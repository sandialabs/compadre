#include "Compadre_FileIO.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_FieldT.hpp"
#include "Compadre_FieldManager.hpp"
#include "Compadre_AnalyticFunctions.hpp"
#include "Compadre_XyzVector.hpp"

#include <sys/stat.h>

#ifdef COMPADREHARNESS_USE_NETCDF
    // NetCDF
    #include <netcdf.h>
#endif

#ifdef COMPADREHARNESS_USE_NETCDF_MPI
    #include <netcdf_par.h>
#endif

namespace Compadre {

typedef Compadre::ParticlesT particles_type;
typedef Compadre::CoordsT coords_type;
typedef Compadre::FieldT field_type;
typedef Compadre::XyzVector xyz_type;

/*
 *
 *  Future improvements: Worry about reading in file with n parts on m processors.
 *
 */

std::string getFilenameExtension(const std::string& filename) {
    size_t pos = filename.rfind('.', filename.length());
    if (pos>=0) {
        return filename.substr(pos+1, filename.length() - pos);
    } else {
        return std::string("No extension found in " + filename + ".");
    }
}

void FileManager::setReader(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const std::string& type) {
    _is_reader = true;
    _particles = particles.getRawPtr();
    _type = type;
    _reader_fn = _fn;
    std::string extension = getFilenameExtension(_reader_fn);
    this->readerIOChoice(extension, false);
}

void FileManager::readerIOChoice(std::string& extension, const bool enforce_serial) {
    transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    // expects lowercase extension

    Teuchos::RCP<Compadre::FileIO> new_io; 
    if (extension == "nc" || extension == "g" || extension == "exo") {
#ifdef COMPADREHARNESS_USE_NETCDF
    #ifdef COMPADREHARNESS_USE_NETCDF_MPI
        // TODO: be careful that parallel reader can actually handle the .nc file. Unless written in netcdf-4 format, it can not.
        // can check style with 'ncdump -k filename'
        // can convert with 'nccopy -k netCDF-4 old_filename new_filename'
        if (enforce_serial) {
            new_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::SerialNetCDFFileIO(_particles)));
        }
        else {
            new_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::ParallelHDF5NetCDFFileIO(_particles)));
        }
    #else
        new_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::SerialNetCDFFileIO(_particles)));
    #endif
#else
        TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with netCDF.");
#endif
    } else {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(true, extension + " is not a supported file type to read from.");
    }
    // if _io is already set, and this object wasn't a writer previously, then copy data
    if (!_io.is_null() && !_is_writer) new_io->copySettingsFrom(*_io); // keeps the state persistent, even if reader changes
    _io = new_io;
    _is_writer = false;
}

void FileManager::setWriter(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const bool use_binary) {
    _is_writer = true;
    _particles = particles.getRawPtr();
    _writer_fn = _fn;
    _use_binary = use_binary;
    std::string extension = getFilenameExtension(_writer_fn);
    this->writerIOChoice(extension, false);
}

void FileManager::writerIOChoice(std::string& extension, const bool enforce_serial) {
    transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    Teuchos::RCP<Compadre::FileIO> new_io; 
    if (extension == "nc" || extension == "g" || extension == "exo") {
#ifdef COMPADREHARNESS_USE_NETCDF
    #ifdef COMPADREHARNESS_USE_NETCDF_MPI
        if (enforce_serial) {
            new_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::SerialNetCDFFileIO(_particles)));
        } else {
              new_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::ParallelHDF5NetCDFFileIO(_particles)));
        }
    #else
        new_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::SerialNetCDFFileIO(_particles)));
    #endif
#else
        TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with netCDF.");
#endif
    } else {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(true, extension + " is not a supported file type to write to.");
    }
    // if _io is already set, and this object wasn't a writer previously, then copy data
    if (!_io.is_null() && !_is_reader) new_io->copySettingsFrom(*_io); // keeps the state persistent, even if reader changes
    _io = new_io;
    _is_reader = false;
}

void FileManager::read() {

    auto ndim_particles = _particles->getCoordsConst()->nDim();
    auto ndim_requested = _particles->getParameters()->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(ndim_particles!=ndim_requested, "Dimension of particles reading in file does not match dimension requested in parameter list.");

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_io.is_null(), "read() called before setReader(...)")
    try {
        _io->read(_reader_fn);
    } catch (int e) {
        if (e==-51 || e==-115) {
            // set the reader again, but this time to use the serial reader
            std::string extension = getFilenameExtension(_reader_fn);
            this->readerIOChoice(extension, true /*enforce serial*/);
            this->read(); // reading will call zoltan2Initialize, so we can just return
            return;
        } else {
            throw e; // rethrow the int because we can't catch this
        }
    }
    _particles->zoltan2Initialize();
}

void FileManager::write() const {
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_io.is_null(), "write() called before setWriter(...)")
    _io->write(_writer_fn, _use_binary);
}


#ifdef COMPADREHARNESS_USE_NETCDF

int SerialNetCDFFileIO::read(const std::string& fn) {
    // TODO: write this to only read in data needed, rather than all and then use what is needed

    // Serial file reader but can be read in serially then split over processors

    // variation of http://www.unidata.ucar.edu/software/netcdf/docs/sfc__pres__temp__rd_8c_source.html
    /* Open the file. */

    // ideally this should only be called when one processor is in the communicator
    // otherwise, it would make more sense to distribute the reading as well as storing the values
    // but this is dealt with by the file manager

    auto ndim_requested = _particles->getParameters()->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_coordinate_layout==CoordinateLayout::LAT_LON_separate && ndim_requested<3, "LAT/LON incompatible with dimension < 3.");

//    int comm_size = _particles->getCoordsConst()->getComm()->getSize();
//        TEUCHOS_TEST_FOR_EXCEPT_MSG(comm_size > 1, "SerialNetCDFFileIO::read called with more than one processor.");

    int ncid, retval;
    retval = nc_open(fn.c_str(), NC_NOWRITE, &ncid);
    TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not opened successfully.");

    /* We will learn about the data file and store results in these
           program variables. */
    int ndims_in, nvars_in, ngatts_in, unlimdimid_in;
    /* There are a number of inquiry functions in netCDF which can be
           used to learn about an unknown netCDF file. NC_INQ tells how
           many netCDF variables, dimensions, and global attributes are in
           the file; also the dimension id of the unlimited dimension, if
           there is one. */
    retval = nc_inq(ncid, &ndims_in, &nvars_in, &ngatts_in, &unlimdimid_in);
    TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not read from successfully.");
    // TEUCHOS_TEST_FOR_EXCEPT_MSG(unlimdimid_in > 0, "Variables with unlimited dimension not currently supported.");

    std::vector<scalar_type> coords_x;
    std::vector<scalar_type> coords_y;
    std::vector<scalar_type> coords_z;
    std::vector<scalar_type> coords_lat;
    std::vector<scalar_type> coords_lon;
    std::vector<local_index_type> flags;
    std::vector<global_index_type> gids;

    // records dimensions-id of dimension with particle numbers
    int particle_num_dimension_id = -1;

    std::vector<bool> identified_fields(nvars_in, false);
    for (local_index_type i=0; i<nvars_in; i++) {
        char var_name[NC_MAX_NAME+1] = "\0";
        retval = nc_inq_varname(ncid, i, var_name);
        std::string var_string_lower(var_name);
        transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
        if (_coordinate_layout==CoordinateLayout::XYZ_separate) {
            if (var_string_lower==_coordinate_names[0]) {
                // get dimension for x and check that it is 1
                int num_dims;
                retval = nc_inq_varndims(ncid, i, &num_dims);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'x'.");

                // store dimensions for this variable
                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                // 
                // Infer particle_num_dimension_id from coordinate, as it is the only dimension that
                // the coordinate variable uses.
                //
                // get dimension id use for specified coordinate name
                particle_num_dimension_id = dims_for_var[0];
                char dim_name[NC_MAX_NAME+1] = "\0";
                // get the name of the dimension
                retval = nc_inq_dimname(ncid, particle_num_dimension_id, dim_name);
                // set the name of the particle number name to this dimension's name
                _particle_num_name = dim_name;

                // first dimension for this variable has number of entries
                size_t num_entries;
                retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
                coords_x.resize(num_entries);

                // read in from netcdf variable
                retval = nc_get_var_double(ncid, i, &coords_x[0]);
                identified_fields[i] = true;
            }
            else if (var_string_lower==_coordinate_names[1]) {
                if (ndim_requested > 1) {
                    int num_dims;
                    retval = nc_inq_varndims(ncid, i, &num_dims);
                    TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'y'.");

                    std::vector<local_index_type> dims_for_var(num_dims,0);
                    retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                    size_t num_entries;
                    retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
                    coords_y.resize(num_entries);

                    retval = nc_get_var_double(ncid, i, &coords_y[0]);
                }
                identified_fields[i] = true;
            }
            else if (var_string_lower==_coordinate_names[2]) {
                if (ndim_requested > 2) {
                    int num_dims;
                    retval = nc_inq_varndims(ncid, i, &num_dims);
                    TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'z'.");

                    std::vector<local_index_type> dims_for_var(num_dims,0);
                    retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                    size_t num_entries;
                    retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
                    coords_z.resize(num_entries);

                    retval = nc_get_var_double(ncid, i, &coords_z[0]);
                }
                identified_fields[i] = true;
            }
        } else if (_coordinate_layout==CoordinateLayout::XYZ_joint) {
            if (var_string_lower==_coordinate_names[0]) {
                // get dimension for x and check that it is 1
                int num_dims;
                retval = nc_inq_varndims(ncid, i, &num_dims);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 2, "Three dimensions should be associated with " + _coordinate_names[0]);

                // store dimensions for this variable
                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                // search for particle num dimension name among the registered dimensions
                int local_particle_num_dimension_id = -1;
                for (int d_id=0; d_id<num_dims; ++d_id) {
                    // get the name of the dimension
                    char dim_name[NC_MAX_NAME+1] = "\0";
                    retval = nc_inq_dimname(ncid, dims_for_var[d_id], dim_name);
                    if (_particle_num_name == dim_name) { // match
                        particle_num_dimension_id = dims_for_var[d_id];
                        local_particle_num_dimension_id = d_id;
                    }
                }
                TEUCHOS_TEST_FOR_EXCEPT_MSG(particle_num_dimension_id < 0, "No dimension name matched: " + _particle_num_name);

                // get particle number dimension
                size_t num_entries;
                retval = nc_inq_dimlen(ncid, dims_for_var[local_particle_num_dimension_id], &num_entries);

                // get size of non-particle number dimensions
                size_t dim_2;
                retval = nc_inq_dimlen(ncid, dims_for_var[(local_particle_num_dimension_id+1)%2], &dim_2);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(dim_2 != 3, "Three dimensions should be associated with " + _coordinate_names[0]);

                // allocate space for xyz array
                auto coords_xyz = std::vector<scalar_type>(num_entries*dim_2);
                coords_x.resize(num_entries);
                if (ndim_requested > 1) coords_y.resize(num_entries);
                if (ndim_requested > 2) coords_z.resize(num_entries);


                // read in from netcdf variable
                retval = nc_get_var_double(ncid, i, &coords_xyz[0]);
                if (local_particle_num_dimension_id==0) {
                    // ordered so particle num dimension is first
                    for (size_t p_num=0; p_num<num_entries; ++p_num) {
                        // copying from (x,y,z,x,y,z,....)
                        coords_x[p_num] = coords_xyz[dim_2*p_num + 0];
                    }
                    if (ndim_requested > 1) {
                        for (size_t p_num=0; p_num<num_entries; ++p_num) {
                            coords_y[p_num] = coords_xyz[dim_2*p_num + 1];
                        }
                    }
                    if (ndim_requested > 2) {
                        for (size_t p_num=0; p_num<num_entries; ++p_num) {
                            coords_z[p_num] = coords_xyz[dim_2*p_num + 2];
                        }
                    }
                } else {
                    // ordered so spatial dimension is first
                    for (size_t p_num=0; p_num<num_entries; ++p_num) {
                        // copying from (x,x,x,x,x,x,....,y,y,y,y,y,y,.....,z,z,z,z,z,z...)
                        coords_x[p_num] = coords_xyz[0*num_entries + p_num];
                    }
                    if (ndim_requested > 1) {
                        for (size_t p_num=0; p_num<num_entries; ++p_num) {
                            coords_y[p_num] = coords_xyz[1*num_entries + p_num];
                        }
                    }
                    if (ndim_requested > 2) {
                        for (size_t p_num=0; p_num<num_entries; ++p_num) {
                            coords_z[p_num] = coords_xyz[2*num_entries + p_num];
                        }
                    }

                }
                identified_fields[i] = true;
            }
        } else if (_coordinate_layout==CoordinateLayout::LAT_LON_separate) {
               if (var_string_lower == _coordinate_names[0]) {
                int num_dims;
                retval = nc_inq_varndims(ncid, i, &num_dims);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with " + _coordinate_names[0]);

                // store dimensions for this variable
                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                // 
                // Infer particle_num_dimension_id from coordinate, as it is the only dimension that
                // the coordinate variable uses.
                //
                // get dimension id use for specified coordinate name
                particle_num_dimension_id = dims_for_var[0];
                char dim_name[NC_MAX_NAME+1] = "\0";
                // get the name of the dimension
                retval = nc_inq_dimname(ncid, particle_num_dimension_id, dim_name);
                // set the name of the particle number name to this dimension's name
                _particle_num_name = dim_name;

                // first dimension for this variable has number of entries
                size_t num_entries;
                retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
                coords_lat.resize(num_entries);

                // read in from netcdf variable
                retval = nc_get_var_double(ncid, i, &coords_lat[0]);
                identified_fields[i] = true;

            } else if (var_string_lower == _coordinate_names[1]) {
                int num_dims;
                retval = nc_inq_varndims(ncid, i, &num_dims);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with " + _coordinate_names[1]);

                // store dimensions for this variable
                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                // first dimension for this variable has number of entries
                size_t num_entries;
                retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
                coords_lon.resize(num_entries);

                // read in from netcdf variable
                retval = nc_get_var_double(ncid, i, &coords_lon[0]);
                identified_fields[i] = true;
            }
        }
    }
    TEUCHOS_TEST_FOR_EXCEPT_MSG(particle_num_dimension_id < 0, "_particle_num_name: " + _particle_num_name + " value not found, or coordinates variable: " + _coordinate_names[0] + " name not found.");

    if (_coordinate_layout==CoordinateLayout::XYZ_joint || _coordinate_layout==CoordinateLayout::XYZ_separate) {
         if (ndim_requested == 2) {
            TEUCHOS_TEST_FOR_EXCEPT_MSG(coords_x.size()!=coords_y.size(), "Different number of " + _coordinate_names[0] + ": " + std::to_string(coords_x.size()) + ", " + _coordinate_names[1] + ": " + std::to_string(coords_y.size()) + "\n");
        }  else if (ndim_requested == 3) {
            TEUCHOS_TEST_FOR_EXCEPT_MSG(coords_x.size()!=coords_y.size() || coords_y.size()!=coords_z.size(), "Different number of " + _coordinate_names[0] + ": " + std::to_string(coords_x.size()) + ", " + _coordinate_names[1] + ": " + std::to_string(coords_y.size()) + ", " + _coordinate_names[2] + ": " + std::to_string(coords_z.size()) + "\n");
        }
    } else if (_coordinate_layout==CoordinateLayout::LAT_LON_separate) {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(coords_lat.size() != coords_lon.size(), _coordinate_names[0] + " size does not match " + _coordinate_names[1] + " size. "
            + std::to_string(coords_lat.size()) + " vs. " + std::to_string(coords_lon.size()) );
        coords_x.resize(coords_lat.size());
        coords_y.resize(coords_lat.size());
        coords_z.resize(coords_lat.size());
    }

    flags.resize(coords_x.size());
    gids.resize(coords_x.size());

    local_index_type flags_var_id = -1;
    local_index_type gids_var_id = -1;

    std::string flag_string_lower(_particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("flags name"));
    transform(flag_string_lower.begin(), flag_string_lower.end(), flag_string_lower.begin(), ::tolower);

    std::string gid_string_lower(_particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("gids name"));
    transform(gid_string_lower.begin(), gid_string_lower.end(), gid_string_lower.begin(), ::tolower);

    for (local_index_type i=0; i<nvars_in; i++) {
        char var_name[NC_MAX_NAME+1] = "\0";
        retval = nc_inq_varname(ncid, i, var_name);
        std::string var_string_lower(var_name);
        transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
        if (var_string_lower==flag_string_lower) {
            retval = nc_get_var_int(ncid, i, &flags[0]);
            identified_fields[i] = true;
            flags_var_id = i;
        }
        else if (var_string_lower==gid_string_lower) {
            retval = nc_get_var_longlong(ncid, i, &gids[0]);
            identified_fields[i] = true;
            gids_var_id = i;
        }
    }

    local_index_type num_fields_identified = 0;
    for (auto val : identified_fields) { if (val) num_fields_identified++; }

    // loop over fields and check whether they have been seen before (x,y,z,flag,id) other-
    // wise read in the variable name, dimension, and values
    local_index_type count = 0;
    std::vector<std::string> field_names(nvars_in - num_fields_identified);
    std::vector<std::string> field_units(nvars_in - num_fields_identified);
    std::vector<bool> field_dim_flipped(nvars_in - num_fields_identified, false);
    std::vector<std::vector<scalar_type> > field_values(nvars_in - num_fields_identified);
    std::vector<size_t> field_dims(nvars_in - num_fields_identified);
    for (local_index_type i=0; i<nvars_in; i++) {
        if (!identified_fields[i]) {
            int num_dims;
            retval = nc_inq_varndims(ncid, i, &num_dims);

               if (num_dims <= 2) {

                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                if (dims_for_var[0]==particle_num_dimension_id)
                    field_dim_flipped[count] = false;
                else if (num_dims>1 && dims_for_var[1]==particle_num_dimension_id)
                    field_dim_flipped[count] = true;
                else {
                    char var_name[NC_MAX_NAME+1] = "\0";
                    retval = nc_inq_varname(ncid, i, var_name);
                    std::cout << "Skipped reading in field: " + std::string(var_name) + " because it has no dimension matching particle_num_dimension_id." << std::endl;
                }

                // hard coded only read in two dimensional data
                size_t dim_1; // size of particle num dimension
                retval = nc_inq_dimlen(ncid, dims_for_var[(field_dim_flipped[count]) ? 1 : 0], &dim_1);
                size_t dim_2 = 1;
                if (num_dims > 1)
                    retval = nc_inq_dimlen(ncid, dims_for_var[(field_dim_flipped[count]) ? 0 : 1], &dim_2);

                if (dim_1 == coords_x.size()) { // otherwise it doesn't match our data sites

                    char var_name[NC_MAX_NAME+1] = "\0";
                    retval = nc_inq_varname(ncid, i, var_name);
                    std::string var_string(var_name);
                    field_names[count] = var_string;

                    size_t unit_name_length;
                    retval = nc_inq_attlen (ncid, i, "units", &unit_name_length);
                    if (retval) unit_name_length = 0;
                    char var2_name[unit_name_length];
                    retval = nc_get_att_text(ncid, i, "units", var2_name);
                    std::string var2_string(var2_name);
                    field_units[count] = var2_string;

                    field_dims[count] = dim_2;
                    // store as 1d array rather than 2d array (use offset i*3+j, etc...)
                    field_values[count] = std::vector<scalar_type>(dim_1*dim_2);

                    retval = nc_get_var_double(ncid, i, &field_values[count][0]);
                    count++;

                } else {
                    char var_name[NC_MAX_NAME+1] = "\0";
                    retval = nc_inq_varname(ncid, i, var_name);
                    std::cout << "Skipped reading in field: " + std::string(var_name) + " because it has no dimension matching particle_num_dimension_id." << std::endl;
                }
            } else {
                char var_name[NC_MAX_NAME+1] = "\0";
                retval = nc_inq_varname(ncid, i, var_name);
                std::cout << "Skipped reading in field: " + std::string(var_name) + " because it is greater than a 2D array." << std::endl;

            }
        }
    } 
    field_names.resize(count);
    field_units.resize(count);
    field_values.resize(count);
    field_dim_flipped.resize(count);
    field_dims.resize(count);

    coords_type* coords = _particles->getCoords();
    auto nPtsGlobal = coords_x.size();

    std::cout << "initializing " << nPtsGlobal << " coordinates...\n";
    std::cout << "initializing map ...\n";

    _particles->resize(nPtsGlobal);
    auto minInd = coords->getMinGlobalIndex();
    auto maxInd = coords->getMaxGlobalIndex();
    auto nPtsLocal = maxInd+1-minInd;

    // to do GID preservation, we first have to get a new coordinate map that breaks up
    // global points, then we get GIDs read in locally, then create a new map using these gids
    if (_particles->getParameters()->get<Teuchos::ParameterList>("io").get<bool>("preserve gids")) {
        // add assert here that gids_var_id >= 0
        TEUCHOS_TEST_FOR_EXCEPT_MSG(gids_var_id < 0, "'gids name' field not found but 'preserve gids' is set to true.");
        auto gids_view = host_view_global_index_type("gids", nPtsLocal, 1);
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nPtsLocal), KOKKOS_LAMBDA(const int i) {
            gids_view(i,0) = gids[i+minInd];
        });
        Kokkos::fence(); 
        _particles->resize(gids_view);
    }

    //std::cout << "min: " << minInd << " max: " << maxInd << std::endl;
    //std::cout << "compare len: " << coords->nLocal() << " to " << maxInd-minInd << std::endl;

    // first fill the coordinates
    host_view_type host_coords = coords->getPts()->getLocalView<host_view_type>(); // assumes we are reading in physical coords in a lagrangian simulation

    if (_coordinate_layout==CoordinateLayout::XYZ_separate || _coordinate_layout==CoordinateLayout::XYZ_joint) {
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nPtsLocal), KOKKOS_LAMBDA(const int i) {
            host_coords(i,0) = coords_x[i+minInd];
            if (ndim_requested > 1) host_coords(i,1) = coords_y[i+minInd];
            if (ndim_requested > 2) host_coords(i,2) = coords_z[i+minInd];
        });
    } else if (_coordinate_layout==CoordinateLayout::LAT_LON_separate) {
           Compadre::CangaSphereTransform sphere_transform(_lat_lon_unit_type=="degrees");
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nPtsLocal), KOKKOS_LAMBDA(const int i) {
            //transform to the sphere based on lat and lon
            xyz_type lat_lon_in(coords_lat[i+minInd], coords_lon[i+minInd], 0);
            xyz_type transformed_lat_lon = sphere_transform.evalVector(lat_lon_in);

            scalar_type coord_norm = std::sqrt(transformed_lat_lon.x*transformed_lat_lon.x + transformed_lat_lon.y*transformed_lat_lon.y + transformed_lat_lon.z*transformed_lat_lon.z);
            host_coords(i,0) = transformed_lat_lon.x / coord_norm;
            host_coords(i,1) = transformed_lat_lon.y / coord_norm;
            host_coords(i,2) = transformed_lat_lon.z / coord_norm;
        });
    }

    // sync coords and fields after the fill
//    coords->syncMemory();
//    coords->print(std::cout);

    if (flags_var_id >= 0) { // only read in if it was identified
        // fill the bc_id
        host_view_local_index_type bc_id = _particles->getFlags()->getLocalView<host_view_local_index_type>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nPtsLocal), KOKKOS_LAMBDA(const int i) {
            bc_id(i,0) = flags[i+minInd];
        });
    }

    for (int i = 0; i < count; i++) {

        _particles->getCoordsConst()->getComm()->barrier();

        // "null" needs update to read in the units
        // field_dims already ordered correctly
        Teuchos::RCP<field_type> field = _particles->getFieldManager()->createField(
                field_dims[i], field_names[i], field_units[i]);

        // fill portion of vector corresponding to global ids that are located on this processor
        host_view_type host_data = field->getLocalVectorVals()->getLocalView<host_view_type>();
        const local_index_type field_dim_i = field_dims[i];

        if (!field_dim_flipped[i]) { // standard dimension ordering (particle numbering first)
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nPtsLocal), KOKKOS_LAMBDA(const int j) {
                // add conditional for index ordering
                for (local_index_type k=0; k<field_dim_i; k++) host_data(j,k) = field_values[i][(j+minInd)*field_dim_i + k];
            });
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nPtsLocal), KOKKOS_LAMBDA(const int j) {
                // add conditional for index ordering
                for (local_index_type k=0; k<field_dim_i; k++) host_data(j,k) = field_values[i][k*coords_x.size() + (j+minInd)];
            });
        }

        field->syncMemory();
    }

    if (_coordinate_layout==CoordinateLayout::LAT_LON_separate && _keep_original_coordinates)    {
        // write original lat and lon to file

        // TODO: "null" needs update to read in the units
        Teuchos::RCP<field_type> old_lat_field = _particles->getFieldManager()->createField(
                1, "original lat", "null");
        Teuchos::RCP<field_type> old_lon_field = _particles->getFieldManager()->createField(
                1, "original lon", "null");

        // fill portion of vector corresponding to global ids that are located on this processor
        host_view_type lat_host_data = old_lat_field->getLocalVectorVals()->getLocalView<host_view_type>();
        host_view_type lon_host_data = old_lon_field->getLocalVectorVals()->getLocalView<host_view_type>();

        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nPtsLocal), KOKKOS_LAMBDA(const int j) {
            lat_host_data(j,0) = coords_lat[j+minInd];
            lon_host_data(j,0) = coords_lon[j+minInd];
        });

        old_lat_field->syncMemory();
        old_lon_field->syncMemory();
    }


    /* Close the file. */
    if ((retval = nc_close(ncid)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not closed successfully.");

    if (this->_particles->getCoords()->isLagrangian()) this->_particles->snapLagrangianCoordsToPhysicalCoords();
    return 1;
}


#ifdef COMPADREHARNESS_USE_NETCDF_MPI

int ParallelHDF5NetCDFFileIO::read(const std::string& fn) {
    #define NC_INDEPENDENT 0
    #define NC_COLLECTIVE 1

    auto ndim_requested = _particles->getParameters()->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_coordinate_layout==CoordinateLayout::LAT_LON_separate && ndim_requested<3, "LAT/LON incompatible with dimension < 3.");

    MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());

    MPI_Info info = MPI_INFO_NULL;

    int ncid, retval;
    retval = nc_open_par(fn.c_str(),
                         NC_NOWRITE|NC_MPIIO,
                         comm,
                         info,
                         &ncid
                         );
    if (retval == -51) {
        std::cout << "File not opened successfully: ERROR:" + std::to_string(retval) + " FILE: " + fn + ". If you are positive that the file exists, check its\ntype with 'ncdump -k yourfile.nc' and verify that it is of type 'netcdf4' and not 'classic'. If it is 'classic', \nrun 'nccopy -k netCDF-4 your_file your_new_file' to convert the style to netCDF-4." << std::endl;
        std::cout << "Switching to serial reader capable of handling classic netCDF files." << std::endl;
        throw -51;
    } else if (retval == -115) {
        throw -115;
    } else {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not opened successfully: ERROR:" + std::to_string(retval) + " FILE: " + fn + ".");
    }

    /* We will learn about the data file and store results in these
           program variables. */
    int ndims_in, nvars_in, ngatts_in, unlimdimid_in;
    /* There are a number of inquiry functions in netCDF which can be
           used to learn about an unknown netCDF file. NC_INQ tells how
           many netCDF variables, dimensions, and global attributes are in
           the file; also the dimension id of the unlimited dimension, if
           there is one. */
    if ((retval = nc_inq(ncid, &ndims_in, &nvars_in, &ngatts_in,
             &unlimdimid_in)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not read from successfully.");
    // TEUCHOS_TEST_FOR_EXCEPT_MSG(unlimdimid_in > 0, "Variables with unlimited dimension not currently supported.");

    size_t coords_x_size, coords_y_size, coords_z_size, coords_lat_size, coords_lon_size;
    std::vector<bool> identified_fields(nvars_in, false);


    // records dimensions-id of dimension with particle numbers
    int particle_num_dimension_id = -1;

    for (local_index_type i=0; i<nvars_in; i++) {
        char var_name[NC_MAX_NAME+1] = "\0";
        retval = nc_inq_varname(ncid, i, var_name);
        std::string var_string_lower(var_name);
        transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
        if (_coordinate_layout==CoordinateLayout::XYZ_separate) {
            if (var_string_lower==_coordinate_names[0]) {
                // get dimension for x and check that it is 1
                int num_dims;
                retval = nc_inq_varndims(ncid, i, &num_dims);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'x'.");

                // store dimensions for this variable
                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                // 
                // Infer particle_num_dimension_id from coordinate, as it is the only dimension that
                // the coordinate variable uses.
                //
                // get dimension id use for specified coordinate name
                particle_num_dimension_id = dims_for_var[0];
                char dim_name[NC_MAX_NAME+1] = "\0";
                // get the name of the dimension
                retval = nc_inq_dimname(ncid, particle_num_dimension_id, dim_name);
                // set the name of the particle number name to this dimension's name
                _particle_num_name = dim_name;

                // first dimension for this variable has number of entries
                retval = nc_inq_dimlen(ncid, dims_for_var[0], &coords_x_size);

                identified_fields[i] = true;
            }
            else if (var_string_lower==_coordinate_names[1]) {
                if (ndim_requested > 1) {
                    int num_dims;
                    retval = nc_inq_varndims(ncid, i, &num_dims);
                    TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'y'.");

                    std::vector<local_index_type> dims_for_var(num_dims,0);
                    retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                    retval = nc_inq_dimlen(ncid, dims_for_var[0], &coords_y_size);
                }
                identified_fields[i] = true;
            }
            else if (var_string_lower==_coordinate_names[2]) {
                if (ndim_requested > 2) {
                    int num_dims;
                    retval = nc_inq_varndims(ncid, i, &num_dims);
                    TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'z'.");

                    std::vector<local_index_type> dims_for_var(num_dims,0);
                    retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                    retval = nc_inq_dimlen(ncid, dims_for_var[0], &coords_z_size);
                }
                identified_fields[i] = true;
            }
        } else if (_coordinate_layout==CoordinateLayout::XYZ_joint) {
            if (var_string_lower==_coordinate_names[0]) {
                // get dimension for x and check that it is 1
                int num_dims;
                retval = nc_inq_varndims(ncid, i, &num_dims);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 2, "Three dimensions should be associated with " + _coordinate_names[0]);

                // store dimensions for this variable
                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                // search for particle num dimension name among the registered dimensions
                int local_particle_num_dimension_id = -1;
                for (int d_id=0; d_id<num_dims; ++d_id) {
                    // get the name of the dimension
                    char dim_name[NC_MAX_NAME+1] = "\0";
                    retval = nc_inq_dimname(ncid, dims_for_var[d_id], dim_name);
                    if (_particle_num_name == dim_name) { // match
                        particle_num_dimension_id = dims_for_var[d_id];
                        local_particle_num_dimension_id = d_id;
                    }
                }
                TEUCHOS_TEST_FOR_EXCEPT_MSG(particle_num_dimension_id < 0, "No dimension name matched: " + _particle_num_name);

                // first dimension for this variable has number of entries
                size_t num_entries;
                retval = nc_inq_dimlen(ncid, dims_for_var[local_particle_num_dimension_id], &num_entries);

                // get size of second dimensions
                size_t dim_2;
                retval = nc_inq_dimlen(ncid, dims_for_var[(local_particle_num_dimension_id+1)%2], &dim_2);

                coords_x_size = num_entries;
                if (ndim_requested > 1) {
                    coords_y_size = num_entries;
                }
                if (ndim_requested > 2) {
                    coords_z_size = num_entries;
                }
                identified_fields[i] = true;
            }
        } else if (_coordinate_layout==CoordinateLayout::LAT_LON_separate) {
            if (var_string_lower==_coordinate_names[0]) {
                // get dimension for x and check that it is 1
                int num_dims;
                retval = nc_inq_varndims(ncid, i, &num_dims);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'lat'.");

                // store dimensions for this variable
                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                // 
                // Infer particle_num_dimension_id from coordinate, as it is the only dimension that
                // the coordinate variable uses.
                //
                // get dimension id use for specified coordinate name
                particle_num_dimension_id = dims_for_var[0];
                char dim_name[NC_MAX_NAME+1] = "\0";
                // get the name of the dimension
                retval = nc_inq_dimname(ncid, particle_num_dimension_id, dim_name);
                // set the name of the particle number name to this dimension's name
                _particle_num_name = dim_name;

                // first dimension for this variable has number of entries
                retval = nc_inq_dimlen(ncid, dims_for_var[0], &coords_lat_size);

                identified_fields[i] = true;
            }
            else if (var_string_lower==_coordinate_names[1]) {
                int num_dims;
                retval = nc_inq_varndims(ncid, i, &num_dims);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'lon'.");

                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                // first dimension for this variable has number of entries
                retval = nc_inq_dimlen(ncid, dims_for_var[0], &coords_lon_size);

                identified_fields[i] = true;
            }
        }
    }

    if (_coordinate_layout==CoordinateLayout::XYZ_joint || _coordinate_layout==CoordinateLayout::XYZ_separate) {
        if (ndim_requested==2) {
            TEUCHOS_TEST_FOR_EXCEPT_MSG(coords_x_size!=coords_y_size, "Different number of " + _coordinate_names[0] + ": " + std::to_string(coords_x_size) + ", " + _coordinate_names[1] + ": " + std::to_string(coords_y_size) + "\n");
        } else if (ndim_requested==3) {
            TEUCHOS_TEST_FOR_EXCEPT_MSG(coords_x_size!=coords_y_size || coords_y_size!=coords_z_size, "Different number of " + _coordinate_names[0] + ": " + std::to_string(coords_x_size) + ", " + _coordinate_names[1] + ": " + std::to_string(coords_y_size) + ", " + _coordinate_names[2] + ": " + std::to_string(coords_z_size) + "\n");
        }
    } else if (_coordinate_layout==CoordinateLayout::LAT_LON_separate) {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(coords_lat_size != coords_lon_size, _coordinate_names[0] + " size does not match " + _coordinate_names[1] + " size. "
            + std::to_string(coords_lat_size) + " vs. " + std::to_string(coords_lon_size) );
        coords_x_size=coords_lat_size;
        coords_y_size=coords_lat_size;
        coords_z_size=coords_lat_size;
    }

    coords_type* coords = _particles->getCoords();
    auto nPtsGlobal = coords_x_size;

    std::cout << "initializing " << nPtsGlobal << " coordinates...\n";
    std::cout << "initializing map ...\n";

    // even if preserving GIDs, need the local sizes to fill with GID information
    _particles->resize(nPtsGlobal);
    auto minInd = coords->getMinGlobalIndex();
    auto maxInd = coords->getMaxGlobalIndex();
    //std::cout << "min: " << minInd << " max: " << maxInd << std::endl;
    //std::cout << "compare len: " << coords->nLocal() << " to " << maxInd-minInd << std::endl;
    local_index_type local_coords_size = (maxInd - minInd + 1);


    std::vector<scalar_type> coords_x(local_coords_size);
    std::vector<scalar_type> coords_y;
    std::vector<scalar_type> coords_z;
    if (ndim_requested > 1) {
        coords_y.resize(local_coords_size);
    }
    if (ndim_requested > 2) {
        coords_z.resize(local_coords_size);
    }
    std::vector<scalar_type> coords_lat;
    std::vector<scalar_type> coords_lon;
    std::vector<local_index_type> flags(local_coords_size);
    std::vector<global_index_type> gids(local_coords_size);

    local_index_type flags_var_id = -1;
    local_index_type gids_var_id = -1;

    std::string flag_string_lower(_particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("flags name"));
    transform(flag_string_lower.begin(), flag_string_lower.end(), flag_string_lower.begin(), ::tolower);

    std::string gid_string_lower(_particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("gids name"));
    transform(gid_string_lower.begin(), gid_string_lower.end(), gid_string_lower.begin(), ::tolower);

    for (local_index_type i=0; i<nvars_in; i++) {
        char var_name[NC_MAX_NAME+1] = "\0";
        retval = nc_inq_varname(ncid, i, var_name);
        std::string var_string_lower(var_name);
        transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);

        if (_coordinate_layout==CoordinateLayout::XYZ_separate) {
            if (var_string_lower==_coordinate_names[0]) {
                unsigned long start = minInd;
                unsigned long countDiff = (unsigned long)(local_coords_size);
                retval = nc_get_vara_double(ncid, i, &start, &countDiff, &coords_x[0]);
            }
            else if (var_string_lower==_coordinate_names[1] && ndim_requested > 1) {
                unsigned long start = minInd;
                unsigned long countDiff = (unsigned long)(local_coords_size);
                retval = nc_get_vara_double(ncid, i, &start, &countDiff, &coords_y[0]);
            }
            else if (var_string_lower==_coordinate_names[2] && ndim_requested > 2) {
                unsigned long start = minInd;
                unsigned long countDiff = (unsigned long)(local_coords_size);
                retval = nc_get_vara_double(ncid, i, &start, &countDiff, &coords_z[0]);
            }
        } else if (_coordinate_layout==CoordinateLayout::XYZ_joint) {
            if (var_string_lower==_coordinate_names[0]) {
                // get dimension for x and check that it is 1
                int num_dims;
                retval = nc_inq_varndims(ncid, i, &num_dims);
                TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 2, "Three dimensions should be associated with " + _coordinate_names[0]);

                // store dimensions for this variable
                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                int local_particle_number_dim_id = (particle_num_dimension_id==dims_for_var[0]) ? 0 : 1;

                // get size of second dimensions (not the particle number dimension)
                size_t dim_2;
                retval = nc_inq_dimlen(ncid, dims_for_var[(local_particle_number_dim_id+1)%2], &dim_2);

                auto coords_xyz = std::vector<scalar_type>(local_coords_size*dim_2);


                unsigned long start[2]; 
                unsigned long countDiff[2]; 
                if (local_particle_number_dim_id==0) { // standard way, particle dim first
                    start[0] = minInd; start[1] = 0;
                    countDiff[0] = (unsigned long)(local_coords_size); countDiff[1] = dim_2;
                } else { // flipped
                    start[0] = 0; start[1] = minInd;
                    countDiff[0] = dim_2; countDiff[1]=(unsigned long)(local_coords_size);

                }

                retval = nc_get_vara_double(ncid, i, start, countDiff, &coords_xyz[0]);

                if (local_particle_number_dim_id==0) {
                    for (int c_num=0; c_num<local_coords_size; ++c_num) {
                        coords_x[c_num] = coords_xyz[dim_2*c_num + 0];
                        if (ndim_requested > 1) coords_y[c_num] = coords_xyz[dim_2*c_num + 1];
                        if (ndim_requested > 2) coords_z[c_num] = coords_xyz[dim_2*c_num + 2];
                    }
                } else { // flipped
                    for (int c_num=0; c_num<local_coords_size; ++c_num) {
                        coords_x[c_num] = coords_xyz[local_coords_size*0 + c_num];
                        if (ndim_requested > 1) coords_y[c_num] = coords_xyz[local_coords_size*1 + c_num];
                        if (ndim_requested > 2) coords_z[c_num] = coords_xyz[local_coords_size*2 + c_num];
                    }
                }
            }
        } else if (_coordinate_layout==CoordinateLayout::LAT_LON_separate) {
            if (var_string_lower==_coordinate_names[0]) {
                unsigned long start = minInd;
                unsigned long countDiff = (unsigned long)(local_coords_size);
                coords_lat.resize(countDiff);
                retval = nc_get_vara_double(ncid, i, &start, &countDiff, &coords_lat[0]);
            }
            else if (var_string_lower==_coordinate_names[1]) {
                unsigned long start = minInd;
                unsigned long countDiff = (unsigned long)(local_coords_size);
                coords_lon.resize(countDiff);
                retval = nc_get_vara_double(ncid, i, &start, &countDiff, &coords_lon[0]);
            }
        }
    }
    for (local_index_type i=0; i<nvars_in; i++) {
        char var_name[NC_MAX_NAME+1] = "\0";
        retval = nc_inq_varname(ncid, i, var_name);
        std::string var_string_lower(var_name);
        transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
        if (var_string_lower==flag_string_lower) {
            unsigned long start = minInd;
            unsigned long countDiff = (unsigned long)(local_coords_size);
            retval = nc_get_vara_int(ncid, i, &start, &countDiff, &flags[0]);
            identified_fields[i] = true;
            flags_var_id = i;
        }
        else if (var_string_lower==gid_string_lower) {
            unsigned long start = minInd;
            unsigned long countDiff = (unsigned long)(local_coords_size);
            retval = nc_get_vara_longlong(ncid, i, &start, &countDiff, &gids[0]);
            identified_fields[i] = true;
            gids_var_id = i;
        }
    }

    // to do GID preservation, we first have to get a new coordinate map that breaks up
    // global points, then we get GIDs read in locally, then create a new map using these gids
    if (_particles->getParameters()->get<Teuchos::ParameterList>("io").get<bool>("preserve gids")) {
        // add assert here that gids_var_id >= 0
        TEUCHOS_TEST_FOR_EXCEPT_MSG(gids_var_id < 0, "'gids name' field not found but 'preserve gids' is set to true.");
        auto gids_view = host_view_global_index_type("gids", local_coords_size, 1);
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int i) {
            gids_view(i,0) = gids[i];
        });
        Kokkos::fence(); 
        _particles->resize(gids_view);
    }

    local_index_type num_fields_identified = 0;
    for (auto val : identified_fields) { if (val) num_fields_identified++; }

    // loop over fields and check whether they have been seen before (x,y,z,flag,id) other-
    // wise read in the variable name, dimension, and values
    local_index_type count = 0;
    std::vector<std::string> field_names(nvars_in - num_fields_identified);
    std::vector<std::string> field_units(nvars_in - num_fields_identified);
    std::vector<bool> field_dim_flipped(nvars_in - num_fields_identified, false);
    std::vector<std::vector<scalar_type> > field_values(nvars_in - num_fields_identified);
    std::vector<size_t> field_dims(nvars_in - num_fields_identified);
    for (local_index_type i=0; i<nvars_in; i++) {
        if (!identified_fields[i]) {

            int num_dims;
            retval = nc_inq_varndims(ncid, i, &num_dims);

               if (num_dims <= 2) {

                std::vector<local_index_type> dims_for_var(num_dims,0);
                retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

                if (dims_for_var[0]==particle_num_dimension_id)
                    field_dim_flipped[count] = false;
                else if (num_dims>1 && dims_for_var[1]==particle_num_dimension_id)
                    field_dim_flipped[count] = true;
                else {
                    char var_name[NC_MAX_NAME+1] = "\0";
                    retval = nc_inq_varname(ncid, i, var_name);
                    std::cout << "Skipped reading in field: " + std::string(var_name) + " because it has no dimension matching particle_num_dimension_id." << std::endl;
                }

                // hard coded only read in two dimensional data
                size_t dim_1; // size of particle num dimension
                retval = nc_inq_dimlen(ncid, dims_for_var[(field_dim_flipped[count]) ? 1 : 0], &dim_1);
                size_t dim_2 = 1;
                if (num_dims > 1)
                    retval = nc_inq_dimlen(ncid, dims_for_var[(field_dim_flipped[count]) ? 0 : 1], &dim_2);

                if (dim_1 == nPtsGlobal) { // otherwise it doesn't match our data sites

                    char var_name[NC_MAX_NAME+1] = "\0";
                    retval = nc_inq_varname(ncid, i, var_name);
                    std::string var_string(var_name);
                    field_names[count] = var_string;

                    size_t unit_name_length;
                    retval = nc_inq_attlen (ncid, i, "units", &unit_name_length);
                    if (retval) unit_name_length = 0;
                    char var2_name[unit_name_length];
                    retval = nc_get_att_text(ncid, i, "units", var2_name);
                    std::string var2_string(var2_name);
                    field_units[count] = var2_string;

                    field_dims[count] = dim_2;
                    // store as 1d array rather than 2d array (use offset i*3+j, etc...)
                    field_values[count] = std::vector<scalar_type>(local_coords_size*dim_2);

                    unsigned long start[2]; 
                    unsigned long countDiff[2]; 
                    if (!field_dim_flipped[count]) { // standard way, particle dim first
                        start[0] = minInd; start[1] = 0;
                        countDiff[0] = (unsigned long)(local_coords_size); countDiff[1] = dim_2;
                    } else { // flipped
                        start[0] = 0; start[1] = minInd;
                        countDiff[0] = dim_2; countDiff[1]=(unsigned long)(local_coords_size);
                    }
                    retval = nc_get_vara_double(ncid, i, start, countDiff, &field_values[count][0]);

                    count++;

                } else {
                    char var_name[NC_MAX_NAME+1] = "\0";
                    retval = nc_inq_varname(ncid, i, var_name);
                    std::cout << "Skipped reading in field: " + std::string(var_name) + " because it has no dimension matching particle_num_dimension_id." << std::endl;
                }
            } else {
                char var_name[NC_MAX_NAME+1] = "\0";
                retval = nc_inq_varname(ncid, i, var_name);
                std::cout << "Skipped reading in field: " + std::string(var_name) + " because it is greater than a 2D array." << std::endl;

            }
        }
    }
    field_names.resize(count);
    field_units.resize(count);
    field_values.resize(count);
    field_dim_flipped.resize(count);
    field_dims.resize(count);

    // first fill the coordinates
    host_view_type host_coords = coords->getPts()->getLocalView<host_view_type>(); // assumes we are reading in physical coords in a lagrangian simulation
    if (_coordinate_layout==CoordinateLayout::XYZ_separate || _coordinate_layout==CoordinateLayout::XYZ_joint) {
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), 
                KOKKOS_LAMBDA(const int i) {
            host_coords(i,0) = coords_x[i];
            if (ndim_requested > 1) host_coords(i,1) = coords_y[i];
            if (ndim_requested > 2) host_coords(i,2) = coords_z[i];
        });
    } else if (_coordinate_layout==CoordinateLayout::LAT_LON_separate) {
           Compadre::CangaSphereTransform sphere_transform(_lat_lon_unit_type=="degrees");
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), 
                KOKKOS_LAMBDA(const int i) {
            //transform to the sphere based on lat and lon
            xyz_type lat_lon_in(coords_lat[i], coords_lon[i], 0);
            xyz_type transformed_lat_lon = sphere_transform.evalVector(lat_lon_in);

            scalar_type coord_norm = std::sqrt(transformed_lat_lon.x*transformed_lat_lon.x + transformed_lat_lon.y*transformed_lat_lon.y + transformed_lat_lon.z*transformed_lat_lon.z);
            host_coords(i,0) = transformed_lat_lon.x / coord_norm;
            host_coords(i,1) = transformed_lat_lon.y / coord_norm;
            host_coords(i,2) = transformed_lat_lon.z / coord_norm;
        });
    }

    // sync coords and fields after the fill
//    coords->syncMemory();
//    coords->print(std::cout);

    if (flags_var_id > 0) { // only read in if it was identified
        // fill the bc_id
        host_view_local_index_type bc_id = _particles->getFlags()->getLocalView<host_view_local_index_type>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int i) {
            bc_id(i,0) = flags[i];
        });
    }

    for (int i = 0; i < count; i++) {

        _particles->getCoordsConst()->getComm()->barrier();

        // "null" needs update to read in the units
        Teuchos::RCP<field_type> field = _particles->getFieldManager()->createField(
                field_dims[i], field_names[i], field_units[i]);

        // fill portion of vector corresponding to global ids that are located on this processor
        host_view_type host_data = field->getLocalVectorVals()->getLocalView<host_view_type>();
        const local_index_type field_dim_i = field_dims[i];
        if (!field_dim_flipped[i]) { // standard dimension ordering (particle numbering first)
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int j) {
                for (local_index_type k=0; k<field_dim_i; k++) host_data(j,k) = field_values[i][j*field_dim_i + k];
            });
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int j) {
                for (local_index_type k=0; k<field_dim_i; k++) host_data(j,k) = field_values[i][k*local_coords_size + j];
            });
        }

        field->syncMemory();
    }

    if (_coordinate_layout==CoordinateLayout::LAT_LON_separate && _keep_original_coordinates)    {
        // write original lat and lon to file

        // TODO: "null" needs update to read in the units
        Teuchos::RCP<field_type> old_lat_field = _particles->getFieldManager()->createField(
                1, "original lat", "null");
        Teuchos::RCP<field_type> old_lon_field = _particles->getFieldManager()->createField(
                1, "original lon", "null");

        // fill portion of vector corresponding to global ids that are located on this processor
        host_view_type lat_host_data = old_lat_field->getLocalVectorVals()->getLocalView<host_view_type>();
        host_view_type lon_host_data = old_lon_field->getLocalVectorVals()->getLocalView<host_view_type>();

        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int j) {
            lat_host_data(j,0) = coords_lat[j];
            lon_host_data(j,0) = coords_lon[j];
        });

        old_lat_field->syncMemory();
        old_lon_field->syncMemory();
    }

    /* Close the file. */
    if ((retval = nc_close(ncid)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not closed successfully.");

    if (this->_particles->getCoords()->isLagrangian()) this->_particles->snapLagrangianCoordsToPhysicalCoords();
    return 1;
}


#endif // COMPADREHARNESS_USE_NETCDF_MPI
#endif // COMPADREHARNESS_USE_NETCDF

#ifdef COMPADREHARNESS_USE_NETCDF

void SerialNetCDFFileIO::write(const std::string& fn, bool use_binary) {

    auto ndim_particles = _particles->getCoordsConst()->nDim();

    //http://www.unidata.ucar.edu/software/netcdf/docs/data_type.html
    //http://www.unidata.ucar.edu/software/netcdf/docs/group__variables.html#ga82d204ebcb895d42d76b780566d91f9a

    int comm_size = _particles->getCoordsConst()->getComm()->getSize();
    TEUCHOS_TEST_FOR_EXCEPT_MSG(comm_size > 1, "SerialNetCDFFileIO::write called with more than one processor.");

    // variation of http://www.unidata.ucar.edu/software/netcdf/docs/sfc__pres__temp__rd_8c_source.html
    // Open the file
    int ncid, retval;
    if ((retval = nc_create(fn.c_str(), NC_NETCDF4, &ncid)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not opened successfully for writing.");

    // register everything in netCDF
    local_index_type particle_dim_id;
    local_index_type x_var_id, y_var_id, z_var_id;

    // first save coords, then save fields
    const coords_type* coords = _particles->getCoordsConst();

    // first get the coordinates
    bool get_physical_coords = !(coords->isLagrangian() && _particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("coordinate type")=="lagrangian");
    // only when both are true do we want the value to be false
    host_view_type host_coords = coords->getPts(false /*halo*/, get_physical_coords)->getLocalView<host_view_type>();
    const local_index_type coords_size = host_coords.extent(0);

    // define the dimensions
    if ((retval = nc_def_dim(ncid, "particle", coords_size, &particle_dim_id)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval!=0, "Error.");

    // define the spatial coordinate variables
    retval = nc_def_var(ncid, "x", NC_DOUBLE, 1, &particle_dim_id, &x_var_id);
    if (ndim_particles > 1) retval = nc_def_var(ncid, "y", NC_DOUBLE, 1, &particle_dim_id, &y_var_id);
    if (ndim_particles > 2) retval = nc_def_var(ncid, "z", NC_DOUBLE, 1, &particle_dim_id, &z_var_id);
    std::string units = this->_particles->getParameters()->get<Teuchos::ParameterList>("coordinates").get<std::string>("units");
    retval = nc_put_att_text(ncid, x_var_id, "units", units.length(), units.c_str());
    if (ndim_particles > 1) retval = nc_put_att_text(ncid, y_var_id, "units", units.length(), units.c_str());
    if (ndim_particles > 2) retval = nc_put_att_text(ncid, z_var_id, "units", units.length(), units.c_str());

    // loop over fields and define for each of them
    const std::vector<Teuchos::RCP<field_type> > fields = _particles->getFieldManagerConst()->getVectorOfFields();
    std::vector<local_index_type> field_dim_ids(fields.size());
    std::vector<local_index_type> field_var_ids(fields.size());
    for (size_t i=0; i<fields.size(); i++){
        size_t comp_size = fields[i]->nDim();
        retval = nc_def_dim(ncid, fields[i]->getName().c_str(), comp_size, &field_dim_ids[i]);

        local_index_type dim_ids[2];
        dim_ids[0] = particle_dim_id;
        dim_ids[1] = field_dim_ids[i];

        retval = nc_def_var(ncid, fields[i]->getName().c_str(), NC_DOUBLE, 2, &dim_ids[0], &field_var_ids[i]);
        retval = nc_put_att_text(ncid, field_var_ids[i], "units", fields[i]->getUnits().length(), fields[i]->getUnits().c_str());
    }

    // fill portion of vector corresponding to global ids that are located on this processor
    local_index_type bc_var_id;
    retval = nc_def_var(ncid, 
            _particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("flags name").c_str(), 
            NC_INT, 1, &particle_dim_id, &bc_var_id);
    retval = nc_put_att_text(ncid, bc_var_id, "units", 4, "none");

    local_index_type gids_var_id = -1;
    if (_particles->getParameters()->get<Teuchos::ParameterList>("io").get<bool>("write gids")) {
        retval = nc_def_var(ncid, 
                _particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("gids name").c_str(), 
                NC_INT64, 1, &particle_dim_id, &gids_var_id);
        retval = nc_put_att_text(ncid, gids_var_id, "units", 4, "none");
    }

    // end of defining dimensions and variables
    retval = nc_enddef(ncid);


    std::vector<scalar_type> coords_x(coords_size);
    std::vector<scalar_type> coords_y, coords_z;
    if (ndim_particles > 1) coords_y.resize(coords_size);
    if (ndim_particles > 2) coords_z.resize(coords_size);
    for (int i=0; i<coords_size; i++) {
        coords_x[i] = host_coords(i,0);
        if (ndim_particles > 1) coords_y[i] = host_coords(i,1);
        if (ndim_particles > 2) coords_z[i] = host_coords(i,2);
    }

    retval = nc_put_var(ncid, x_var_id, &coords_x[0]);
    if (ndim_particles > 1) retval = nc_put_var(ncid, y_var_id, &coords_y[0]);
    if (ndim_particles > 2) retval = nc_put_var(ncid, z_var_id, &coords_z[0]);

    for (size_t i=0; i<fields.size(); i++){

        std::vector<scalar_type> host_field_data_vec(coords_size*fields[i]->nDim());

        host_view_type host_field_data = fields[i]->getLocalVectorVals()->getLocalView<host_view_type>();
//        {
//            // ONLY FOR TESTING!!!!!
//            for (int j=0; j<3; j++) {
//                for (int k=0; k<coords_size; k++) {
//                    host_field_data(k,j) = j;
//                }
//            }
//        }
        const local_index_type field_i_ndim = fields[i]->nDim();
        for (int j=0; j<coords_size; j++) {
            for (int k=0; k<field_i_ndim; k++) {
                host_field_data_vec[field_i_ndim*j+k] = host_field_data(j,k);
            }
        }
        retval = nc_put_var(ncid, field_var_ids[i], &host_field_data_vec[0]);
    }


    {
        // flags
        host_view_local_index_type bc_id = _particles->getFlags()->getLocalView<host_view_local_index_type>();
        std::vector<local_index_type> bc_id_vec(coords_size);
        for (int k=0; k<coords_size; k++) {
            bc_id_vec[k] = bc_id(k,0);
        }
        retval = nc_put_var(ncid, bc_var_id, &bc_id_vec[0]);


        if (_particles->getParameters()->get<Teuchos::ParameterList>("io").get<bool>("write gids")) {
            // GIDs
            auto gids = _particles->getCoordsConst()->getMapConst()->getMyGlobalIndices();

            std::vector<global_index_type> gids_vec(coords_size);
            for (int k=0; k<coords_size; k++) {
                gids_vec[k] = gids(k);
            }

            retval = nc_put_var_longlong(ncid, gids_var_id, &gids_vec[0]);
        }

    }

    // Close the file
    if ((retval = nc_close(ncid)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not closed successfully.");
}

#ifdef COMPADREHARNESS_USE_NETCDF_MPI

void ParallelHDF5NetCDFFileIO::write(const std::string& fn, bool use_binary) {

    auto ndim_particles = _particles->getCoordsConst()->nDim();

    #define NC_INDEPENDENT 0
    #define NC_COLLECTIVE 1

    MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
 
    MPI_Info info = MPI_INFO_NULL;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int ncid, retval;
    if (( retval = nc_create_par(fn.c_str(),
                                 NC_NETCDF4|NC_MPIIO,
                                 comm,
                                 info,
                                 &ncid
                                 )))

        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not created successfully.");

//    std::cout << "rank " << rank << " of " << size << std::endl;

    // register everything in netCDF
    local_index_type particle_dim_id;
    local_index_type x_var_id, y_var_id, z_var_id;

    // first save coords, then save fields
    const coords_type* coords = _particles->getCoordsConst();
    global_index_type global_coords_size = coords->nGlobal();

    // first get the coordinates
    bool get_physical_coords = !(coords->isLagrangian() && _particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("coordinate type")=="lagrangian");
    // only when both are true do we want the value to be false
    host_view_type host_coords = coords->getPts(false /*halo*/, get_physical_coords)->getLocalView<host_view_type>();
    const local_index_type coords_size = host_coords.extent(0);

    // define the dimensions
    if ((retval = nc_def_dim(ncid, "particle", global_coords_size, &particle_dim_id)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval!=0, "Error.");

    // define the spatial coordinate variables
    retval = nc_def_var(ncid, "x", NC_DOUBLE, 1, &particle_dim_id, &x_var_id);
    if (ndim_particles > 1) retval = nc_def_var(ncid, "y", NC_DOUBLE, 1, &particle_dim_id, &y_var_id);
    if (ndim_particles > 2) retval = nc_def_var(ncid, "z", NC_DOUBLE, 1, &particle_dim_id, &z_var_id);
    std::string units = this->_particles->getParameters()->get<Teuchos::ParameterList>("coordinates").get<std::string>("units");
    retval = nc_put_att_text(ncid, x_var_id, "units", units.length(), units.c_str());
    if (ndim_particles > 1) retval = nc_put_att_text(ncid, y_var_id, "units", units.length(), units.c_str());
    if (ndim_particles > 2) retval = nc_put_att_text(ncid, z_var_id, "units", units.length(), units.c_str());

    // loop over fields and define for each of them
    const std::vector<Teuchos::RCP<field_type> > fields = _particles->getFieldManagerConst()->getVectorOfFields();
    std::vector<local_index_type> field_dim_ids(fields.size());
    std::vector<local_index_type> field_var_ids(fields.size());
    for (size_t i=0; i<fields.size(); i++){
        size_t comp_size = fields[i]->nDim();
        if (comp_size > 1) {
            retval = nc_def_dim(ncid, fields[i]->getName().c_str(), comp_size, &field_dim_ids[i]);

            local_index_type dim_ids[2];
            dim_ids[0] = particle_dim_id;
            dim_ids[1] = field_dim_ids[i];

            retval = nc_def_var(ncid, fields[i]->getName().c_str(), NC_DOUBLE, 2, &dim_ids[0], &field_var_ids[i]);
            retval = nc_put_att_text(ncid, field_var_ids[i], "units", fields[i]->getUnits().length(), fields[i]->getUnits().c_str());
        } else if (comp_size==1) {
            local_index_type dim_id = particle_dim_id;
            retval = nc_def_var(ncid, fields[i]->getName().c_str(), NC_DOUBLE, 1, &dim_id, &field_var_ids[i]);
            retval = nc_put_att_text(ncid, field_var_ids[i], "units", fields[i]->getUnits().length(), fields[i]->getUnits().c_str());
        }
    }

    // fill portion of vector corresponding to global ids that are located on this processor
    local_index_type bc_var_id;
    retval = nc_def_var(ncid, 
            _particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("flags name").c_str(), 
            NC_INT, 1, &particle_dim_id, &bc_var_id);
    if ((retval = nc_put_att_text(ncid, bc_var_id, "units", 4, "none")))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "flag units didn't finish.");

    local_index_type gids_var_id = -1;
    if (_particles->getParameters()->get<Teuchos::ParameterList>("io").get<bool>("write gids")) {
        retval = nc_def_var(ncid, 
                _particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("gids name").c_str(), 
                NC_INT64, 1, &particle_dim_id, &gids_var_id);
        if (!retval) {
            retval = nc_put_att_text(ncid, gids_var_id, "units", 4, "none");
        } else if (retval==-42) { // name already in use
            retval = nc_inq_varid(ncid, 
                    _particles->getParameters()->get<Teuchos::ParameterList>("io").get<std::string>("gids name").c_str(), 
                    &gids_var_id);
            retval = nc_put_att_text(ncid, gids_var_id, "units", 4, "none");
        }
    }
    //printf("retval: %d\n", retval);
    //if ((retval = nc_put_att_text(ncid, ids_var_id, "units", 4, "none")))
    //    TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "id units didn't finish.");

    // end of defining dimensions and variables
    if ((retval = nc_enddef(ncid)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "definitions didn't finish.");

//    std::cout << coords_size << " on " << rank << std::endl;
    std::vector<scalar_type> coords_x(coords_size);
    std::vector<scalar_type> coords_y, coords_z;
    if (ndim_particles > 1) coords_y.resize(coords_size);
    if (ndim_particles > 2) coords_z.resize(coords_size);
    for (int i=0; i<coords_size; i++) {
        coords_x[i] = host_coords(i,0);
        if (ndim_particles > 1) coords_y[i] = host_coords(i,1);
        if (ndim_particles > 2) coords_z[i] = host_coords(i,2);
    }

    if ((retval = nc_var_par_access(ncid, x_var_id, NC_COLLECTIVE)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "x write permission not changed.");
    if (ndim_particles > 1) {
        if ((retval = nc_var_par_access(ncid, y_var_id, NC_COLLECTIVE)))
            TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "y write permission not changed.");
    }
    if (ndim_particles > 2) {
        if ((retval = nc_var_par_access(ncid, z_var_id, NC_COLLECTIVE)))
            TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "z write permission not changed.");
    }

    // this mapping temporarily gives the min and max elements which is equivalent to an offset
    Teuchos::RCP<map_type> temporary_map = Teuchos::rcp(new map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), coords_size, 0, _particles->getCoordsConst()->getComm()));

    {
        unsigned long start = (unsigned long)(temporary_map->getMinGlobalIndex());
        unsigned long countDiff = (unsigned long)(coords_size);

        if ((retval = nc_put_vara_double(ncid, x_var_id, &start, &countDiff, &coords_x[0])))
            TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "x not written.");
        if (ndim_particles > 1) {
            if ((retval = nc_put_vara_double(ncid, y_var_id, &start, &countDiff, &coords_y[0])))
                TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "y not written.");
        }
        if (ndim_particles > 2) {
            if ((retval = nc_put_vara_double(ncid, z_var_id, &start, &countDiff, &coords_z[0])))
                TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "z not written.");
        }
    }


    //bool IDs_handled = false;
    for (size_t i=0; i<fields.size(); i++){

        std::vector<scalar_type> host_field_data_vec(coords_size*fields[i]->nDim());

        host_view_type host_field_data = fields[i]->getLocalVectorVals()->getLocalView<host_view_type>();
//        {
//            // ONLY FOR TESTING!!!!!
//            for (int j=0; j<3; j++) {
//                for (int k=0; k<coords_size; k++) {
//                    host_field_data(k,j) = j;
//                }
//            }
//        }
        const local_index_type field_i_ndim = fields[i]->nDim();
        for (int j=0; j<coords_size; j++) {
            for (int k=0; k<field_i_ndim; k++) {
                host_field_data_vec[field_i_ndim*j+k] = host_field_data(j,k);
            }
        }

        unsigned long start[2]; start[0] = (unsigned long)(temporary_map->getMinGlobalIndex()); start[1] = 0;
        unsigned long countDiff[2]; countDiff[0] = (unsigned long)(coords_size); countDiff[1] = fields[i]->nDim();
        retval = nc_put_vara(ncid, field_var_ids[i], start, countDiff, &host_field_data_vec[0]);

    }
    {
        unsigned long start = (unsigned long)(temporary_map->getMinGlobalIndex());
        unsigned long countDiff = (unsigned long)(coords_size);

        // flags
        host_view_local_index_type bc_id = _particles->getFlags()->getLocalView<host_view_local_index_type>();
        std::vector<local_index_type> bc_id_vec(coords_size);
        for (int k=0; k<coords_size; k++) {
            bc_id_vec[k] = bc_id(k,0);
        }
        retval = nc_put_vara(ncid, bc_var_id, &start, &countDiff, &bc_id_vec[0]);

        if (_particles->getParameters()->get<Teuchos::ParameterList>("io").get<bool>("write gids")) {
            // GIDs
            auto gids = _particles->getCoordsConst()->getMapConst()->getMyGlobalIndices();

            std::vector<global_index_type> gids_vec(coords_size);
            for (int k=0; k<coords_size; k++) {
                gids_vec[k] = gids(k);
            }
            retval = nc_put_vara_longlong(ncid, gids_var_id, &start, &countDiff, &gids_vec[0]);
        }
    }

    // Close the file
    if ((retval = nc_close(ncid)))
        TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not closed successfully.");
}

#endif // COMPADREHARNESS_USE_NETCDF
#endif // COMPADREHARNESS_USE_NETCDF_MPI

}
