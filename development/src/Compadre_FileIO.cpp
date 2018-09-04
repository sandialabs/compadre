#include "Compadre_FileIO.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_FieldT.hpp"
#include "Compadre_FieldManager.hpp"
#include "Compadre_AnalyticFunctions.hpp"
#include "Compadre_XyzVector.hpp"

#ifdef COMPADRE_USE_VTK

	// VTK Fundamentals
	#include <vtkStreamingDemandDrivenPipeline.h>
	#include <vtkDataSetAttributes.h>
	#include <vtkInformation.h>
	#include <vtkPointData.h>
	#include <vtkDoubleArray.h>
	#include <vtkPolyData.h>
	#include <vtkUnstructuredGrid.h>

	// VTK Reader / Writers
	#include <vtkPolyDataReader.h>
	#include <vtkPDataSetWriter.h>
	#include <vtkXMLPPolyDataWriter.h>
	#include <vtkXMLPPolyDataReader.h>
	#include <vtkXMLPUnstructuredGridWriter.h>
	#include <vtkXMLPUnstructuredGridReader.h>

	// VTK Filters
	#include <vtkAppendFilter.h>
	#include <vtkDelaunay3D.h>
	#include <vtkDistributedDataFilter.h>
	#include <vtkVertexGlyphFilter.h>

	// VTK MPI
	#include <vtkMPIController.h>
	#include <vtkMPI.h>

#endif

#ifdef COMPADRE_USE_NETCDF
	// NetCDF
	#include <netcdf.h>
#endif

#ifdef COMPADRE_USE_NETCDF_MPI
	#include <netcdf_par.h>
#endif

//#include <vtkXMLMultiBlockDataReader.h>
//#include <vtkMultiBlockDataSet.h>
//#include <vtkDataSetMapper.h>

//#include <vtkPExodusIIReader.h>
//#include <vtkPExodusIIWriter.h>

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

void FileManager::setReader(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const std::string& type, const bool keep_original_lat_lon) {
	_particles = particles.getRawPtr();
	_type = type;
	_reader_fn = _fn;
	// get extension
	size_t pos = _reader_fn.rfind('.', _reader_fn.length());
	std::string extension = _reader_fn.substr(pos+1, _reader_fn.length() - pos);
	transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

	std::string type_to_lower = _type;
	transform(type_to_lower.begin(), type_to_lower.end(), type_to_lower.begin(), ::tolower);

	if (extension == "vtk" || extension == "vtu" || extension == "vtp") {
#ifdef COMPADRE_USE_VTK
		_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::LegacyVTKFileIO(particles.getRawPtr()))); // data being read in is just polydata
		_is_parallel = false;
#else
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with VTK.");
#endif
	} else if (extension == "pvtp") {
#ifdef COMPADRE_USE_VTK
		_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::XMLVTPFileIO(particles.getRawPtr())));
		_is_parallel = true;
#else
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with VTK.");
#endif
	} else if (extension == "pvtu") {
#ifdef COMPADRE_USE_VTK
		_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::XMLVTUFileIO(particles.getRawPtr())));
		_is_parallel = true;
#else
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with VTK.");
#endif
	} else if (extension == "nc") {
#ifdef COMPADRE_USE_NETCDF
	#ifdef COMPADRE_USE_NETCDF_MPI
		// TODO: be careful that Parallel reader can actually handle the .nc file. Unless written by a parallel writer, it can not.
		if (type_to_lower.length() == 0) {
			_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::ParallelHDF5NetCDFFileIO(particles.getRawPtr())));
		} else if (type_to_lower == "homme") {
//			_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::ParallelHOMMEFileIO(particles.getRawPtr())));
			_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::SerialHOMMEFileIO(particles.getRawPtr(), keep_original_lat_lon)));
		}
		_is_parallel = true;
	#else
		if (type_to_lower == "homme") {
			_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::SerialHOMMEFileIO(particles.getRawPtr(), keep_original_lat_lon)));
		} else {
			_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::SerialNetCDFFileIO(particles.getRawPtr())));
		}
		_is_parallel = false;
	#endif
#else
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with netCDF.");
#endif
	} else {
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, extension + " is not a supported file type to read from.");
	}
}

void FileManager::setWriter(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const bool use_binary) {
	_writer_fn = _fn;
	_use_binary = use_binary;

	// get extension
	size_t pos = _writer_fn.rfind('.', _writer_fn.length());
	std::string extension = _writer_fn.substr(pos+1, _writer_fn.length() - pos);
	transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
	if (extension == "pvtp") {
#ifdef COMPADRE_USE_VTK
		_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::XMLVTPFileIO(particles.getRawPtr())));
#else
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with VTK.");
#endif
	} else if (extension == "pvtu") {
#ifdef COMPADRE_USE_VTK
		_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::XMLVTUFileIO(particles.getRawPtr())));
#else
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with VTK.");
#endif
	} else if (extension == "pvtk") {
#ifdef COMPADRE_USE_VTK
		_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::LegacyVTKFileIO(particles.getRawPtr())));
#else
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with VTK.");
#endif
	} else if (extension == "nc") {
#ifdef COMPADRE_USE_NETCDF
	#ifdef COMPADRE_USE_NETCDF_MPI
			_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::ParallelHDF5NetCDFFileIO(particles.getRawPtr())));
	#else
			_io = Teuchos::rcp_static_cast<Compadre::FileIO>(Teuchos::rcp(new Compadre::SerialNetCDFFileIO(particles.getRawPtr())));
	#endif
#else
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not built with netCDF.");
#endif
	} else {
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, extension + " is not a supported file type to write to.");
	}
}

void FileManager::read() const {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_io.is_null(), "read() called before setReader(...)")
	int retval = _io->read(_reader_fn);
	// could add a test here for failure in parallel mode and switch to a serial reader over n processors
	// (each processor must read all data, but only store around 1/n of it)
	// TODO: such serial readers are not yet written
//	if (retval>0 && _is_parallel) // error
//		this->setReader(_reader_fn, _particles, _type, true /* enforce serial */);
	_particles->zoltan2Initialize();
}

void FileManager::write() const {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_io.is_null(), "write() called before setWriter(...)")
	_io->write(_writer_fn, _use_binary);
}

void FileManager::generateWriteMesh() {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_io.is_null(), "generateMesh() called before setWriter(...)")
	// get extension
	size_t pos = _writer_fn.rfind('.', _writer_fn.length());
	std::string extension = _writer_fn.substr(pos+1, _writer_fn.length() - pos);
	transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

	// get base of _writer_fn
	std::string _writer_fn_base = _writer_fn.substr(0,pos);
	if (extension == "pvtu" || extension == "pvtp") {
		_io->generateMesh();
		_io->writeMesh(_writer_fn_base+"_mesh.pvtu");
	} else {
		std::cout << "WARNING: generateMesh() called on invalid filetype != .pvtu, generateMesh() not executed." << std::endl;
	}
}

#ifdef COMPADRE_USE_VTK

void VTKData::generateDataSet(bool include_halo, bool for_writing_output, bool use_physical_coords) {

	vtkSmartPointer<vtkPoints> points =
		vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkPolyData> polydata =
		vtkSmartPointer<vtkPolyData>::New();

	vtkSmartPointer<vtkPoints> ghost_points;
	vtkSmartPointer<vtkPolyData> ghost_polydata;

	if (include_halo) {
		ghost_points = vtkSmartPointer<vtkPoints>::New();
		ghost_polydata = vtkSmartPointer<vtkPolyData>::New();
	}

	// first save coords, then save fields
	const coords_type* coords = _particles->getCoordsConst();

	// first fill the coordinates
	// if Lagrangian simulation:
	//   if parameter deck indicates Lagrangian, we want a flag of false
	//   if use_physical_coords is false, we want a flag of false
	//   otherwise true

	// TODO: this combination of output desired for writing and input used for neighborhoods must be split
	bool get_physical_coords = true;
	if (for_writing_output) get_physical_coords = (_parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate type")=="physical");
	else get_physical_coords = use_physical_coords;
//	if (coords->isLagrangian()) {
//		if (_parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate type")=="lagrangian") get_physical_coords = false;
//		if (!use_physical_coords) get_physical_coords = false;
//	}

	// only when both are true do we want the value to be false
	device_view_type dev_coords = coords->getPts(false /*halo*/, get_physical_coords)->getLocalView<device_view_type>();
	device_view_type dev_coords_halo_only;
	if (include_halo) dev_coords_halo_only = coords->getPts(true /*halo*/, get_physical_coords)->getLocalView<device_view_type>();

	const local_index_type coords_size = dev_coords.dimension_0();
	const local_index_type halo_coords_size = dev_coords_halo_only.dimension_0();

	const local_index_type ndim = coords->nDim();
	points->SetNumberOfPoints(coords_size);
	if (include_halo) ghost_points->SetNumberOfPoints(halo_coords_size);
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,coords_size), KOKKOS_LAMBDA(const int i) {
		double coord[ndim];
		for (local_index_type j=0; j<coords->nDim(); j++) coord[j] = dev_coords(i,j);
		points->SetPoint(i, coord);
	});
	if (include_halo) {
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,halo_coords_size), KOKKOS_LAMBDA(const int i) {
			double coord[ndim];
			for (local_index_type j=0; j<coords->nDim(); j++) coord[j] = dev_coords_halo_only(i,j);
			ghost_points->SetPoint(i, coord);
		});
	}
	polydata->SetPoints(points);
	if (include_halo) ghost_polydata->SetPoints(ghost_points);

	const std::vector<Teuchos::RCP<field_type> > fields = _particles->getFieldManagerConst()->getVectorOfFields();

	for (local_index_type i=0; i<fields.size(); i++){
		vtkSmartPointer<vtkDoubleArray> field_data =
			vtkSmartPointer<vtkDoubleArray>::New();
		vtkSmartPointer<vtkDoubleArray> ghost_field_data;
		if (include_halo) ghost_field_data = vtkSmartPointer<vtkDoubleArray>::New();

		const vtkIdType comp_size = fields[i]->nDim();
		field_data->SetNumberOfComponents(comp_size);
		field_data->SetName(fields[i]->getName().c_str());

		if (include_halo) {
			ghost_field_data->SetNumberOfComponents(comp_size);
			ghost_field_data->SetName(fields[i]->getName().c_str());
		}


		// fill portion of vector corresponding to global ids that are located on this processor
		host_view_type host_field_data = fields[i]->getLocalVectorVals()->getLocalView<host_view_type>();
		host_view_type host_field_data_halo_only;
		if (include_halo) host_field_data_halo_only = fields[i]->getLocalVectorVals(true /*halo*/)->getLocalView<host_view_type>();

		field_data->SetNumberOfTuples(host_field_data.dimension_0());
		if (include_halo) ghost_field_data->SetNumberOfTuples(host_field_data_halo_only.dimension_0());
		const local_index_type field_data_size = host_field_data.dimension_0();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,field_data_size), KOKKOS_LAMBDA(const int j) {
			double data[comp_size];
			for (local_index_type k=0; k<comp_size; k++) data[k] = host_field_data(j,k);
			field_data->SetTuple(j, data);
		});
		if (include_halo) {
			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,host_field_data_halo_only.dimension_0()), KOKKOS_LAMBDA(const int j) {
				double data[comp_size];
				for (local_index_type k=0; k<comp_size; k++) data[k] = host_field_data_halo_only(j,k);
				ghost_field_data->SetTuple(j, data);
			});
		}
		if (include_halo) ghost_polydata->GetPointData()->AddArray(ghost_field_data);
		polydata->GetPointData()->AddArray(field_data);
	}
	{
		vtkSmartPointer<vtkDoubleArray> bc_id_data =
			vtkSmartPointer<vtkDoubleArray>::New();
		vtkSmartPointer<vtkDoubleArray> ghost_bc_id_data;
		if (include_halo) ghost_bc_id_data = vtkSmartPointer<vtkDoubleArray>::New();

		const vtkIdType comp_size = 1;
		bc_id_data->SetNumberOfComponents(comp_size);
		bc_id_data->SetName("FLAG");

		if (include_halo) {
			ghost_bc_id_data->SetNumberOfComponents(comp_size);
			ghost_bc_id_data->SetName("FLAG");
		}


		// fill portion of vector corresponding to global ids that are located on this processor
		bc_id_data->SetNumberOfTuples(_particles->getFlags()->getLocalLength());
		if (include_halo) ghost_bc_id_data->SetNumberOfTuples(halo_coords_size);
		host_view_type bc_id = _particles->getFlags()->getLocalView<host_view_type>();
		const local_index_type flag_size = bc_id.dimension_0();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,flag_size), KOKKOS_LAMBDA(const int j) {
			double data[comp_size];
			data[0] = bc_id(j,0);
			bc_id_data->SetTuple(j, data);
		});
		const local_index_type my_rank = _particles->getCoordsConst()->getComm()->getRank();
		if (include_halo) {
			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,halo_coords_size), KOKKOS_LAMBDA(const int j) {
				double data[comp_size];
				data[0] = -my_rank;
				ghost_bc_id_data->SetTuple(j, data);
			});
		}
		if (include_halo) ghost_polydata->GetPointData()->AddArray(ghost_bc_id_data);
		polydata->GetPointData()->AddArray(bc_id_data);
	}
	{
		vtkSmartPointer<vtkDoubleArray> ids =
			vtkSmartPointer<vtkDoubleArray>::New();
		vtkSmartPointer<vtkDoubleArray> ghost_ids;
		if (include_halo) ghost_ids = vtkSmartPointer<vtkDoubleArray>::New();

		ids->SetNumberOfComponents(1);
		ids->SetName("ID");

		if (include_halo) {
			ghost_ids->SetNumberOfComponents(1);
			ghost_ids->SetName("ID");
		}


		// fill portion of vector corresponding to global ids that are located on this processor
		ids->SetNumberOfTuples(_particles->getCoordsConst()->nLocal());
		if (include_halo) ghost_ids->SetNumberOfTuples(halo_coords_size);

		typedef Kokkos::View<const global_index_type*> const_gid_view_type;
		const_gid_view_type gids = _particles->getCoordsConst()->getMapConst()->getMyGlobalIndices();
		const_gid_view_type ghost_gids;
		if (include_halo) ghost_gids = _particles->getCoordsConst()->getMapConst(true /*halo*/)->getMyGlobalIndices();

		const local_index_type id_size = gids.dimension_0();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,id_size), KOKKOS_LAMBDA(const int j) {
			double data[1];
			data[0] = gids(j,0);
			ids->SetTuple(j, data);
		});
		if (include_halo) {
			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,halo_coords_size), KOKKOS_LAMBDA(const int j) {
				double data[1];
				data[0] = ghost_gids(j,0);
				ghost_ids->SetTuple(j, data);
			});
		}
		if (include_halo) ghost_polydata->GetPointData()->AddArray(ghost_ids);
		polydata->GetPointData()->AddArray(ids);
	}

	_dataSet = polydata;
	if (include_halo) _haloDataSet = ghost_polydata;

}

void VTKData::generateCombinedDataSet() {
	vtkSmartPointer<vtkAppendFilter> appendFilter =
		vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter->AddInputData(_dataSet);
	appendFilter->AddInputData(_haloDataSet);
	appendFilter->Update();
	_data_w_halo_DataSet = vtkDataSet::SafeDownCast(appendFilter->GetOutput());
}


void VTKFileIO::populateParticles() {

	coords_type* coords = _particles->getCoords();

	nPtsLocal = _dataSet->GetNumberOfPoints();

//			std::cout << "initializing " << nPtsLocal << " coordinates locally...\n";
//						std::cout << "initializing map ...\n";

	_particles->resize(nPtsLocal, true /*local number known resize*/);
	nPtsGlobal = coords->getMapConst()->getGlobalNumElements();

	// Now check for point data
	vtkPointData *pd = _dataSet->GetPointData();
	if (pd)
	{
//				std::cout << nPtsGlobal << " points contains point data with "
//					<< pd->GetNumberOfArrays() << " "
//					<< pd->GetNumberOfComponents()
//					<< " arrays." << std::endl;

		// global indices but are local to this processor
		minInd = coords->getMinGlobalIndex();
		maxInd = coords->getMaxGlobalIndex();

//				std::cout << "min: " << minInd << " max: " << maxInd << std::endl;

		// first fill the coordinates
		host_view_type dev_coords = coords->getPts()->getLocalView<host_view_type>(); // assumes we are reading in physical coords in a lagrangian simulation

		const local_index_type ndim = coords->nDim();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,maxInd+1-minInd), KOKKOS_LAMBDA(const int i) {
			double coord[ndim];
			_dataSet->GetPoint(i, coord);
			for (local_index_type j=0; j<coords->nDim(); j++) dev_coords(i,j) = coord[j];
		});

		// sync coords and fields after the fill
//		coords->syncMemory();
//				coords->print(std::cout);

		for (int i = 0; i < pd->GetNumberOfArrays(); i++) {
			_particles->getCoordsConst()->getComm()->barrier();
//					std::cout << "\tArray " << i
//					<< " is named "
//					<< (pd->GetArrayName(i) ? pd->GetArrayName(i) : "NULL")
//					<< " with "
//					<< std::endl;

			const vtkIdType comp_size = pd->GetArray(i)->GetNumberOfComponents();
			std::string lowerCaseArrayName = pd->GetArrayName(i);
			transform(lowerCaseArrayName.begin(), lowerCaseArrayName.end(), lowerCaseArrayName.begin(), ::tolower);
			if ( std::strcmp(lowerCaseArrayName.c_str(), "bc_id") == 0  || std::strcmp(lowerCaseArrayName.c_str(), "flag") == 0 ) {
				host_view_type bc_id = _particles->getFlags()->getLocalView<host_view_type>();
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,maxInd+1-minInd), KOKKOS_LAMBDA(const int j) {
					double data[comp_size];
					pd->GetArray(i)->GetTuple(j, data);
					bc_id(j,0) = data[0];
				});
			} else {
				Teuchos::RCP<field_type> field = _particles->getFieldManager()->createField(
						comp_size, (pd->GetArrayName(i) ? pd->GetArrayName(i) : "NULL"), "null");

				// fill portion of vector corresponding to global ids that are located on this processor
				device_view_type dev_data = field->getLocalVectorVals()->getLocalView<device_view_type>();
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,maxInd+1-minInd), KOKKOS_LAMBDA(const int j) {
					double data[comp_size];
					pd->GetArray(i)->GetTuple(j, data);
					for (local_index_type k=0; k<comp_size; k++) dev_data(j,k) = data[k];
				});
				field->syncMemory();
			}
		}
	}
}

#endif // COMPADRE_USE_VTK

#ifdef COMPADRE_USE_NETCDF

int SerialNetCDFFileIO::read(const std::string& fn) {
	// TODO: write this to only read in data needed, rather than all and then use what is needed

	// Serial file reader but can be read in serially then split over processors

	// variation of http://www.unidata.ucar.edu/software/netcdf/docs/sfc__pres__temp__rd_8c_source.html
	/* Open the file. */

	// ideally this should only be called when one processor is in the communicator
	// otherwise, it would make more sense to distribute the reading as well as storing the values
	// but this is dealt with by the file manager

//	int comm_size = _particles->getCoordsConst()->getComm()->getSize();
//		TEUCHOS_TEST_FOR_EXCEPT_MSG(comm_size > 1, "SerialNetCDFFileIO::read called with more than one processor.");

	int ncid, retval;
	if ((retval = nc_open(fn.c_str(), NC_NOWRITE, &ncid)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not opened successfully.");

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

	std::vector<scalar_type> coords_x;
	std::vector<scalar_type> coords_y;
	std::vector<scalar_type> coords_z;
	std::vector<local_index_type> flags;
	std::vector<global_index_type> ids;

	std::vector<bool> identified_fields(nvars_in, false);

	for (local_index_type i=0; i<nvars_in; i++) {
		char var_name[256];
		retval = nc_inq_varname(ncid, i, var_name);
		std::string var_string_lower(var_name);
		transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
		if (var_string_lower=="x") {
			// get dimension for x and check that it is 1
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'x'.");

			// store dimensions for this variable
			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			// first dimension for this variable has number of entries
			size_t num_entries;
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
			coords_x.resize(num_entries);

			// read in from netcdf variable
			retval = nc_get_var_double(ncid, i, &coords_x[0]);
			identified_fields[i] = true;
		}
		else if (var_string_lower=="y") {
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'y'.");

			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			size_t num_entries;
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
			coords_y.resize(num_entries);

			retval = nc_get_var_double(ncid, i, &coords_y[0]);
			identified_fields[i] = true;
		}
		else if (var_string_lower=="z") {
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'z'.");

			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			size_t num_entries;
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
			coords_z.resize(num_entries);

			retval = nc_get_var_double(ncid, i, &coords_z[0]);
			identified_fields[i] = true;
		}
	}

	TEUCHOS_TEST_FOR_EXCEPT_MSG(coords_x.size()!=coords_y.size() || coords_y.size()!=coords_z.size(), "Different number of x, y, and z coordinates.");
	flags.resize(coords_x.size());
	local_index_type flags_var_id = -1, ids_var_id = -1;

	ids.resize(coords_x.size());
	for (local_index_type i=0; i<nvars_in; i++) {
		char var_name[256];
		retval = nc_inq_varname(ncid, i, var_name);
		std::string var_string_lower(var_name);
		transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
		if (var_string_lower=="flag") {
			retval = nc_get_var_int(ncid, i, &flags[0]);
			identified_fields[i] = true;
			flags_var_id = i;
		}
		else if (var_string_lower=="id") {
			retval = nc_get_var_long(ncid, i, &ids[0]);
			identified_fields[i] = true;
			ids_var_id = i;
		}
	}

	local_index_type num_fields_identified = 0;
	for (auto val : identified_fields) { if (val) num_fields_identified++; }

	// loop over fields and check whether they have been seen before (x,y,z,flag,id) other-
	// wise read in the variable name, dimension, and values
	local_index_type count = 0;
	std::vector<std::string> field_names(nvars_in - num_fields_identified);
	std::vector<std::vector<scalar_type> > field_values(nvars_in - num_fields_identified);
	std::vector<size_t> field_dims(nvars_in - num_fields_identified);
	for (local_index_type i=0; i<nvars_in; i++) {
		if (!identified_fields[i]) {
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);

			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			size_t dim_1;
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &dim_1);

			size_t dim_2 = 1;
			if (num_dims > 1)
				retval = nc_inq_dimlen(ncid, dims_for_var[1], &dim_2);

			char var_name[256];
			retval = nc_inq_varname(ncid, i, var_name);
			std::string var_string(var_name);

			field_names[count] = var_string;
			field_dims[count] = dim_2;
			// store as 1d array rather than 2d array (use offset i*3+j, etc...)
			field_values[count] = std::vector<scalar_type>(dim_1*dim_2);

			retval = nc_get_var_double(ncid, i, &field_values[count][0]);
			count++;
		}
	}

	coords_type* coords = _particles->getCoords();
	nPtsGlobal = coords_x.size();

	std::cout << "initializing " << nPtsGlobal << " coordinates...\n";
	std::cout << "initializing map ...\n";

	_particles->resize(nPtsGlobal);

	minInd = coords->getMinGlobalIndex();
	maxInd = coords->getMaxGlobalIndex();

	std::cout << "min: " << minInd << " max: " << maxInd << std::endl;
	std::cout << "compare len: " << coords->nLocal() << " to " << maxInd-minInd << std::endl;

	// first fill the coordinates
	host_view_type host_coords = coords->getPts()->getLocalView<host_view_type>(); // assumes we are reading in physical coords in a lagrangian simulation
	const local_index_type ndim = coords->nDim();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int i) {
		host_coords(i-minInd,0) = coords_x[i];
		host_coords(i-minInd,1) = coords_y[i];
		host_coords(i-minInd,2) = coords_z[i];
	});

	// sync coords and fields after the fill
//	coords->syncMemory();
//	coords->print(std::cout);

	if (flags_var_id > 0) { // only read in if it was identified
		// fill the bc_id
		host_view_type bc_id = _particles->getFlags()->getLocalView<host_view_type>();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int i) {
			bc_id(i-minInd,0) = flags[i];
		});
	}

	for (int i = 0; i < count; i++) {

		_particles->getCoordsConst()->getComm()->barrier();

		// "null" needs update to read in the units
		Teuchos::RCP<field_type> field = _particles->getFieldManager()->createField(
				field_dims[i], field_names[i], "null");

		// fill portion of vector corresponding to global ids that are located on this processor
		host_view_type host_data = field->getLocalVectorVals()->getLocalView<host_view_type>();
		const local_index_type field_dim_i = field_dims[i];
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int j) {
			for (local_index_type k=0; k<field_dim_i; k++) host_data(j-minInd,k) = field_values[i][j*field_dim_i + k];
		});

		field->syncMemory();
	}

	/* Close the file. */
	if ((retval = nc_close(ncid)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not closed successfully.");

	if (this->_particles->getCoords()->isLagrangian()) this->_particles->snapLagrangianCoordsToPhysicalCoords();
	return 1;
}

int SerialHOMMEFileIO::read(const std::string& fn) {
	// TODO: write this to only read in data needed, rather than all and then use what is needed

	// Serial file reader but can be read in serially then split over processors

	// variation of http://www.unidata.ucar.edu/software/netcdf/docs/sfc__pres__temp__rd_8c_source.html
	/* Open the file. */

	// ideally this should only be called when one processor is in the communicator
	// otherwise, it would make more sense to distribute the reading as well as storing the values
	// but this is dealt with by the file manager

//	int comm_size = _particles->getCoordsConst()->getComm()->getSize();
//		TEUCHOS_TEST_FOR_EXCEPT_MSG(comm_size > 1, "SerialNetCDFFileIO::read called with more than one processor.");

	int ncid, retval;
	if ((retval = nc_open(fn.c_str(), NC_NOWRITE, &ncid)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not opened successfully.");

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

	// need to get dimensions for ncol or grid_size and grid_corners or ncorners

	int file_type = 0;
	// type 1: gives ncorners, which must be traversed as ncol to get the lat and lon
	// type 2: gives grid_center_lat and grid_center_lon which can be used directly
	for (local_index_type i=0; i<ndims_in; i++) {
		char dim_name[256];
		retval = nc_inq_dimname(ncid, i, dim_name);
		std::string dim_string_lower(dim_name);
		transform(dim_string_lower.begin(), dim_string_lower.end(), dim_string_lower.begin(), ::tolower);
		if (dim_string_lower=="ncorners") {
			file_type = 1;
			break;
		} else if (dim_string_lower=="grid_corners") {
			file_type = 2;
			break;
		}
	}
	TEUCHOS_TEST_FOR_EXCEPT_MSG(file_type==0, "File dimensions not as expected.");


	std::string lat_variable_name;
	std::string lon_variable_name;
	if (file_type == 1) {
		lat_variable_name = "lat";
		lon_variable_name = "lon";
	} else if (file_type == 2) {
		lat_variable_name = "grid_center_lat";
		lon_variable_name = "grid_center_lon";
	}

	std::vector<scalar_type> coords_x, coords_y, coords_z, lat_vals, lon_vals;
	std::vector<local_index_type> flags;
	std::vector<global_index_type> ids;

	std::vector<bool> identified_fields(nvars_in, false);

	bool in_degrees = false;

	for (local_index_type i=0; i<nvars_in; i++) {
		char var_name[256];
		retval = nc_inq_varname(ncid, i, var_name);
		std::string var_string_lower(var_name);
		transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
		if (var_string_lower == lat_variable_name) {
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'lat'.");

			// store dimensions for this variable
			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			// first dimension for this variable has number of entries
			size_t num_entries;
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
			lat_vals.resize(num_entries);

			// read in from netcdf variable
			retval = nc_get_var_double(ncid, i, &lat_vals[0]);
			identified_fields[i] = true;

			// check if in degrees or radians
			size_t unit_name_length;
			retval = nc_inq_attlen (ncid, i, "units", &unit_name_length);
			if (retval) unit_name_length = 0;
			char var_name[unit_name_length];
			retval = nc_get_att_text(ncid, i, "units", var_name);
			std::string var_string(var_name);
			if (var_string.substr(0,3) == "deg") in_degrees = true;
			else in_degrees = false;
		} else if (var_string_lower == lon_variable_name) {
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'lon'.");

			// store dimensions for this variable
			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			// first dimension for this variable has number of entries
			size_t num_entries;
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &num_entries);
			lon_vals.resize(num_entries);

			// read in from netcdf variable
			retval = nc_get_var_double(ncid, i, &lon_vals[0]);
			identified_fields[i] = true;
		}
	}

	TEUCHOS_TEST_FOR_EXCEPT_MSG(lat_vals.size() != lon_vals.size(), "lat size does not match lon size. "
			+ std::to_string(lat_vals.size()) + " vs. " + std::to_string(lon_vals.size()) );
	coords_x.resize(lat_vals.size());
	coords_y.resize(lat_vals.size());
	coords_z.resize(lat_vals.size());


	flags.resize(coords_x.size());
	local_index_type flags_var_id = -1, ids_var_id = -1;

	ids.resize(coords_x.size());
	for (local_index_type i=0; i<nvars_in; i++) {
		char var_name[256];
		retval = nc_inq_varname(ncid, i, var_name);
		std::string var_string_lower(var_name);
		transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
		if (var_string_lower=="flag") {
			retval = nc_get_var_int(ncid, i, &flags[0]);
			identified_fields[i] = true;
			flags_var_id = i;
		}
		else if (var_string_lower=="id") {
			retval = nc_get_var_long(ncid, i, &ids[0]);
			identified_fields[i] = true;
			ids_var_id = i;
		}
	}


	local_index_type num_fields_identified = 0;
	for (auto val : identified_fields) { if (val) num_fields_identified++; }

	// loop over fields and check whether they have been seen before (x,y,z,flag,id) other-
	// wise read in the variable name, dimension, and values
	local_index_type count = 0;
	std::vector<std::string> field_names(nvars_in - num_fields_identified);
	std::vector<std::string> field_units(nvars_in - num_fields_identified);
	std::vector<std::vector<scalar_type> > field_values(nvars_in - num_fields_identified);
	std::vector<size_t> field_dims(nvars_in - num_fields_identified);
	for (local_index_type i=0; i<nvars_in; i++) {
		if (!identified_fields[i]) {
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);

			if (num_dims <= 2) {
				std::vector<local_index_type> dims_for_var(num_dims,0);
				retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

				// hard coded only read in two dimensional data
				size_t dim_1;
				retval = nc_inq_dimlen(ncid, dims_for_var[0], &dim_1);

				if (dim_1 == coords_x.size()) { // otherwise it doesn't match our data sites
					size_t dim_2 = 1;
					if (num_dims > 1)
						retval = nc_inq_dimlen(ncid, dims_for_var[1], &dim_2);

					char var_name[256];
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
				}
			}
		}
	}
	field_names.resize(count);
	field_units.resize(count);
	field_values.resize(count);
	field_dims.resize(count);


	coords_type* coords = _particles->getCoords();
	nPtsGlobal = coords_x.size();

	std::cout << "initializing " << nPtsGlobal << " coordinates...\n";
	std::cout << "initializing map ...\n";

	_particles->resize(nPtsGlobal);

	minInd = coords->getMinGlobalIndex();
	maxInd = coords->getMaxGlobalIndex();

	std::cout << "min: " << minInd << " max: " << maxInd << std::endl;
	std::cout << "compare len: " << coords->nLocal() << " to " << maxInd-minInd << std::endl;

	// first fill the coordinates
	host_view_type host_coords = coords->getPts()->getLocalView<host_view_type>(); // assumes we are reading in physical coords in a lagrangian simulation
	const local_index_type ndim = coords->nDim();
	Compadre::CangaSphereTransform sphere_transform(in_degrees);
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int i) {
		//transform to the sphere based on lat and lon
		xyz_type lat_lon_in(lat_vals[i], lon_vals[i], 0);
		xyz_type transformed_lat_lon = sphere_transform.evalVector(lat_lon_in);

		scalar_type coord_norm = std::sqrt(transformed_lat_lon.x*transformed_lat_lon.x + transformed_lat_lon.y*transformed_lat_lon.y + transformed_lat_lon.z*transformed_lat_lon.z);

		host_coords(i-minInd,0) = transformed_lat_lon.x / coord_norm;
		host_coords(i-minInd,1) = transformed_lat_lon.y / coord_norm;
		host_coords(i-minInd,2) = transformed_lat_lon.z / coord_norm;
	});

	// sync coords and fields after the fill
//	coords->syncMemory();
//	coords->print(std::cout);

	if (flags_var_id > 0) { // only read in if it was identified
		// fill the bc_id
		host_view_type bc_id = _particles->getFlags()->getLocalView<host_view_type>();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int i) {
			bc_id(i-minInd,0) = flags[i];
		});
	}

	for (int i = 0; i < count; i++) {

		_particles->getCoordsConst()->getComm()->barrier();

		// TODO: "null" needs update to read in the units
		Teuchos::RCP<field_type> field = _particles->getFieldManager()->createField(
				field_dims[i], field_names[i], field_units[i]);

		// fill portion of vector corresponding to global ids that are located on this processor
		host_view_type host_data = field->getLocalVectorVals()->getLocalView<host_view_type>();
		const local_index_type field_dim_i = field_dims[i];
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int j) {
			for (local_index_type k=0; k<field_dim_i; k++) host_data(j-minInd,k) = field_values[i][j*field_dim_i + k];
		});

		field->syncMemory();
	}


	if (_keep_original_lat_lon)	{
		// write original lat and lon to file

		// TODO: "null" needs update to read in the units
		Teuchos::RCP<field_type> old_lat_field = _particles->getFieldManager()->createField(
				1, "original lat", "null");
		Teuchos::RCP<field_type> old_lon_field = _particles->getFieldManager()->createField(
				1, "original lon", "null");

		// fill portion of vector corresponding to global ids that are located on this processor
		host_view_type lat_host_data = old_lat_field->getLocalVectorVals()->getLocalView<host_view_type>();
		host_view_type lon_host_data = old_lon_field->getLocalVectorVals()->getLocalView<host_view_type>();

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int j) {
			lat_host_data(j-minInd,0) = lat_vals[j];
			lon_host_data(j-minInd,0) = lon_vals[j];
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

#ifdef COMPADRE_USE_NETCDF_MPI

int ParallelHDF5NetCDFFileIO::read(const std::string& fn) {
	#define NC_INDEPENDENT 0
	#define NC_COLLECTIVE 1

	MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
	MPI_Info info = MPI_INFO_NULL;

	int ncid, retval;
	if (( retval = nc_open_par (    fn.c_str(),
									NC_NOWRITE|NC_MPIIO,
									comm,
									info,
									&ncid
									)))

		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not opened successfully." + std::to_string(retval));

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

	size_t coords_x_size, coords_y_size, coords_z_size;
	std::vector<bool> identified_fields(nvars_in, false);

	for (local_index_type i=0; i<nvars_in; i++) {
		char var_name[256];
		retval = nc_inq_varname(ncid, i, var_name);
		std::string var_string_lower(var_name);
		transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
		if (var_string_lower=="x") {
			// get dimension for x and check that it is 1
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'x'.");

			// store dimensions for this variable
			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			// first dimension for this variable has number of entries
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &coords_x_size);

			identified_fields[i] = true;
		}
		else if (var_string_lower=="y") {
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'y'.");

			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			retval = nc_inq_dimlen(ncid, dims_for_var[0], &coords_y_size);

			identified_fields[i] = true;
		}
		else if (var_string_lower=="z") {
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'z'.");

			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			retval = nc_inq_dimlen(ncid, dims_for_var[0], &coords_z_size);

			identified_fields[i] = true;
		}
	}


	TEUCHOS_TEST_FOR_EXCEPT_MSG(coords_x_size!=coords_y_size || coords_y_size!=coords_z_size, "Different number of global x, y, and z coordinates.");

	coords_type* coords = _particles->getCoords();
	nPtsGlobal = coords_x_size;

	std::cout << "initializing " << nPtsGlobal << " coordinates...\n";
	std::cout << "initializing map ...\n";

	_particles->resize(nPtsGlobal);

	minInd = coords->getMinGlobalIndex();
	maxInd = coords->getMaxGlobalIndex();

	std::cout << "min: " << minInd << " max: " << maxInd << std::endl;
	std::cout << "compare len: " << coords->nLocal() << " to " << maxInd-minInd << std::endl;

	local_index_type local_coords_size = (maxInd - minInd + 1);


	std::vector<scalar_type> coords_x(local_coords_size);
	std::vector<scalar_type> coords_y(local_coords_size);
	std::vector<scalar_type> coords_z(local_coords_size);
	std::vector<local_index_type> flags(local_coords_size);
	std::vector<global_index_type> ids(local_coords_size);

	local_index_type flags_var_id = -1, ids_var_id = -1;

	for (local_index_type i=0; i<nvars_in; i++) {
		char var_name[256];
		retval = nc_inq_varname(ncid, i, var_name);
		std::string var_string_lower(var_name);
		transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
		if (var_string_lower=="x") {
			unsigned long start = minInd;
			unsigned long countDiff = (unsigned long)(local_coords_size);
			retval = nc_get_vara_double(ncid, i, &start, &countDiff, &coords_x[0]);
		}
		else if (var_string_lower=="y") {
			unsigned long start = minInd;
			unsigned long countDiff = (unsigned long)(local_coords_size);
			retval = nc_get_vara_double(ncid, i, &start, &countDiff, &coords_y[0]);
		}
		else if (var_string_lower=="z") {
			unsigned long start = minInd;
			unsigned long countDiff = (unsigned long)(local_coords_size);
			retval = nc_get_vara_double(ncid, i, &start, &countDiff, &coords_z[0]);
		}
		else if (var_string_lower=="flag") {
			unsigned long start = minInd;
			unsigned long countDiff = (unsigned long)(local_coords_size);
			retval = nc_get_vara_int(ncid, i, &start, &countDiff, &flags[0]);
			identified_fields[i] = true;
			flags_var_id = i;
		}
		else if (var_string_lower=="id") {
			unsigned long start = minInd;
			unsigned long countDiff = (unsigned long)(local_coords_size);
			retval = nc_get_vara_long(ncid, i, &start, &countDiff, &ids[0]);
			identified_fields[i] = true;
			ids_var_id = i;
		}
	}

	local_index_type num_fields_identified = 0;
	for (auto val : identified_fields) { if (val) num_fields_identified++; }

	// loop over fields and check whether they have been seen before (x,y,z,flag,id) other-
	// wise read in the variable name, dimension, and values
	local_index_type count = 0;
	std::vector<std::string> field_names(nvars_in - num_fields_identified);
	std::vector<std::vector<scalar_type> > field_values(nvars_in - num_fields_identified);
	std::vector<size_t> field_dims(nvars_in - num_fields_identified);
	for (local_index_type i=0; i<nvars_in; i++) {
		if (!identified_fields[i]) {

			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);

			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			size_t dim_1;
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &dim_1);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(dim_1 == local_coords_size, "Wrong dimensions for field data.");

			size_t dim_2 = 1;
			if (num_dims > 1)
				retval = nc_inq_dimlen(ncid, dims_for_var[1], &dim_2);

			char var_name[256];
			retval = nc_inq_varname(ncid, i, var_name);
			std::string var_string(var_name);

			field_names[count] = var_string;
			field_dims[count] = dim_2;
			// store as 1d array rather than 2d array (use offset i*3+j, etc...)
			field_values[count] = std::vector<scalar_type>(local_coords_size*dim_2);

			unsigned long start[2]; start[0] = minInd; start[1] = 0;
			unsigned long countDiff[2]; countDiff[0] = (unsigned long)(local_coords_size); countDiff[1] = dim_2;
			retval = nc_get_vara_double(ncid, i, start, countDiff, &field_values[count][0]);//field_var_ids[i], start, countDiff, &host_field_data_vec[0]);

			//retval = nc_get_var_double(ncid, i, &field_values[count][0]);
			count++;
		}
	}

	// first fill the coordinates
	host_view_type host_coords = coords->getPts()->getLocalView<host_view_type>(); // assumes we are reading in physical coords in a lagrangian simulation
	const local_index_type ndim = coords->nDim();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int i) {
		host_coords(i,0) = coords_x[i];
		host_coords(i,1) = coords_y[i];
		host_coords(i,2) = coords_z[i];
	});

	// sync coords and fields after the fill
//	coords->syncMemory();
//	coords->print(std::cout);

	if (flags_var_id > 0) { // only read in if it was identified
		// fill the bc_id
		host_view_type bc_id = _particles->getFlags()->getLocalView<host_view_type>();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int i) {
			bc_id(i,0) = flags[i];
		});
	}

	for (int i = 0; i < count; i++) {

		_particles->getCoordsConst()->getComm()->barrier();

		// "null" needs update to read in the units
		Teuchos::RCP<field_type> field = _particles->getFieldManager()->createField(
				field_dims[i], field_names[i], "null");

		// fill portion of vector corresponding to global ids that are located on this processor
		host_view_type host_data = field->getLocalVectorVals()->getLocalView<host_view_type>();
		const local_index_type field_dim_i = field_dims[i];
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int j) {
			for (local_index_type k=0; k<field_dim_i; k++) host_data(j,k) = field_values[i][j*field_dim_i + k];
		});

		field->syncMemory();
	}

	/* Close the file. */
	if ((retval = nc_close(ncid)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not closed successfully.");

	if (this->_particles->getCoords()->isLagrangian()) this->_particles->snapLagrangianCoordsToPhysicalCoords();
	return 1;
}

int ParallelHOMMEFileIO::read(const std::string& fn) {
	#define NC_INDEPENDENT 0
	#define NC_COLLECTIVE 1

	MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
	MPI_Info info = MPI_INFO_NULL;

	int ncid, retval;
	if (( retval = nc_open_par (    fn.c_str(),
									NC_NOWRITE|NC_MPIIO,
									comm,
									info,
									&ncid
									)))

		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not opened successfully.");

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

	// need to get dimensions for ncol or grid_size and grid_corners or ncorners

	int file_type = 0;
	// type 1: gives ncorners, which must be traversed as ncol to get the lat and lon
	// type 2: gives grid_center_lat and grid_center_lon which can be used directly
	for (local_index_type i=0; i<ndims_in; i++) {
		char dim_name[256];
		retval = nc_inq_dimname(ncid, i, dim_name);
		std::string dim_string_lower(dim_name);
		transform(dim_string_lower.begin(), dim_string_lower.end(), dim_string_lower.begin(), ::tolower);
		if (dim_string_lower=="ncorners") {
			file_type = 1;
			break;
		} else if (dim_string_lower=="grid_corners") {
			file_type = 2;
			break;
		}
	}
	TEUCHOS_TEST_FOR_EXCEPT_MSG(file_type==0, "File dimensions not as expected.");


	std::string lat_variable_name;
	std::string lon_variable_name;
	if (file_type == 1) {
		lat_variable_name = "lat";
		lon_variable_name = "lon";
	} else if (file_type == 2) {
		lat_variable_name = "grid_center_lat";
		lon_variable_name = "grid_center_long";
	}

	size_t coords_x_size, coords_y_size, coords_z_size, lat_size, lon_size;
	std::vector<bool> identified_fields(nvars_in, false);

	for (local_index_type i=0; i<nvars_in; i++) {
		char var_name[256];
		retval = nc_inq_varname(ncid, i, var_name);
		std::string var_string_lower(var_name);
		transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
		if (var_string_lower == lat_variable_name) {
			// get dimension for x and check that it is 1
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'lat'.");

			// store dimensions for this variable
			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			// first dimension for this variable has number of entries
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &lat_size);

			identified_fields[i] = true;
		}
		else if (var_string_lower== lon_variable_name) {
			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_dims != 1, "Only one dimension should be associated with 'lon'.");

			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			// first dimension for this variable has number of entries
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &lon_size);

			identified_fields[i] = true;
		}
	}
	TEUCHOS_TEST_FOR_EXCEPT_MSG(lat_size != lon_size, "lat size does not match lon size.");
	coords_x_size = lat_size;
	coords_y_size = lat_size;
	coords_z_size = lat_size;

	coords_type* coords = _particles->getCoords();
	nPtsGlobal = coords_x_size;

	std::cout << "initializing " << nPtsGlobal << " coordinates...\n";
	std::cout << "initializing map ...\n";

	_particles->resize(nPtsGlobal);

	minInd = coords->getMinGlobalIndex();
	maxInd = coords->getMaxGlobalIndex();

	std::cout << "min: " << minInd << " max: " << maxInd << std::endl;
	std::cout << "compare len: " << coords->nLocal() << " to " << maxInd-minInd << std::endl;

	local_index_type local_coords_size = (maxInd - minInd + 1);

	std::vector<scalar_type> lat_vals(local_coords_size);
	std::vector<scalar_type> lon_vals(local_coords_size);

	std::vector<scalar_type> coords_x(local_coords_size);
	std::vector<scalar_type> coords_y(local_coords_size);
	std::vector<scalar_type> coords_z(local_coords_size);

	std::vector<local_index_type> flags(local_coords_size);
	std::vector<global_index_type> ids(local_coords_size);

	local_index_type flags_var_id = -1, ids_var_id = -1;

	for (local_index_type i=0; i<nvars_in; i++) {
		char var_name[256];
		retval = nc_inq_varname(ncid, i, var_name);
		std::string var_string_lower(var_name);
		transform(var_string_lower.begin(), var_string_lower.end(), var_string_lower.begin(), ::tolower);
		if (var_string_lower == lat_variable_name) {
			unsigned long start = minInd;
			unsigned long countDiff = (unsigned long)(local_coords_size);
			retval = nc_get_vara_double(ncid, i, &start, &countDiff, &lat_vals[0]);
		}
		else if (var_string_lower == lon_variable_name) {
			unsigned long start = minInd;
			unsigned long countDiff = (unsigned long)(local_coords_size);
			retval = nc_get_vara_double(ncid, i, &start, &countDiff, &lon_vals[0]);
		}
		else if (var_string_lower=="flag") {
			unsigned long start = minInd;
			unsigned long countDiff = (unsigned long)(local_coords_size);
			retval = nc_get_vara_int(ncid, i, &start, &countDiff, &flags[0]);
			identified_fields[i] = true;
			flags_var_id = i;
		}
		else if (var_string_lower=="id") {
			unsigned long start = minInd;
			unsigned long countDiff = (unsigned long)(local_coords_size);
			retval = nc_get_vara_long(ncid, i, &start, &countDiff, &ids[0]);
			identified_fields[i] = true;
			ids_var_id = i;
		}
	}

	local_index_type num_fields_identified = 0;
	for (auto val : identified_fields) { if (val) num_fields_identified++; }

	// loop over fields and check whether they have been seen before (x,y,z,flag,id) other-
	// wise read in the variable name, dimension, and values
	local_index_type count = 0;
	std::vector<std::string> field_names(nvars_in - num_fields_identified);
	std::vector<std::vector<scalar_type> > field_values(nvars_in - num_fields_identified);
	std::vector<size_t> field_dims(nvars_in - num_fields_identified);
	for (local_index_type i=0; i<nvars_in; i++) {
		if (!identified_fields[i]) {

			int num_dims;
			retval = nc_inq_varndims(ncid, i, &num_dims);

			std::vector<local_index_type> dims_for_var(num_dims,0);
			retval = nc_inq_vardimid(ncid, i, &dims_for_var[0]);

			size_t dim_1;
			retval = nc_inq_dimlen(ncid, dims_for_var[0], &dim_1);
			TEUCHOS_TEST_FOR_EXCEPT_MSG(dim_1 == local_coords_size, "Wrong dimensions for field data.");

			size_t dim_2 = 1;
			if (num_dims > 1)
				retval = nc_inq_dimlen(ncid, dims_for_var[1], &dim_2);

			char var_name[256];
			retval = nc_inq_varname(ncid, i, var_name);
			std::string var_string(var_name);

			field_names[count] = var_string;
			field_dims[count] = dim_2;
			// store as 1d array rather than 2d array (use offset i*3+j, etc...)
			field_values[count] = std::vector<scalar_type>(local_coords_size*dim_2);

			unsigned long start[2]; start[0] = minInd; start[1] = 0;
			unsigned long countDiff[2]; countDiff[0] = (unsigned long)(local_coords_size); countDiff[1] = dim_2;
			retval = nc_get_vara_double(ncid, i, start, countDiff, &field_values[count][0]);//field_var_ids[i], start, countDiff, &host_field_data_vec[0]);

			//retval = nc_get_var_double(ncid, i, &field_values[count][0]);
			count++;
		}
	}

	// first fill the coordinates
	host_view_type host_coords = coords->getPts()->getLocalView<host_view_type>(); // assumes we are reading in physical coords in a lagrangian simulation
	const local_index_type ndim = coords->nDim();
	Compadre::CangaSphereTransform sphere_transform;
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int i) {
		//transform to the sphere based on lat and lon
		xyz_type lat_lon_in(lat_vals[i], lon_vals[i], 0);
		xyz_type transformed_lat_lon = sphere_transform.evalVector(lat_lon_in);

		scalar_type coord_norm = std::sqrt(transformed_lat_lon.x*transformed_lat_lon.x + transformed_lat_lon.y*transformed_lat_lon.y + transformed_lat_lon.z*transformed_lat_lon.z);

		host_coords(i,0) = transformed_lat_lon.x / coord_norm;
		host_coords(i,1) = transformed_lat_lon.y / coord_norm;
		host_coords(i,2) = transformed_lat_lon.z / coord_norm;
	});

	// sync coords and fields after the fill
//	coords->syncMemory();
//	coords->print(std::cout);

	if (flags_var_id > 0) { // only read in if it was identified
		// fill the bc_id
		host_view_type bc_id = _particles->getFlags()->getLocalView<host_view_type>();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int i) {
			bc_id(i,0) = flags[i];
		});
	}

	for (int i = 0; i < count; i++) {

		_particles->getCoordsConst()->getComm()->barrier();

		// "null" needs update to read in the units
		Teuchos::RCP<field_type> field = _particles->getFieldManager()->createField(
				field_dims[i], field_names[i], "null");

		// fill portion of vector corresponding to global ids that are located on this processor
		host_view_type host_data = field->getLocalVectorVals()->getLocalView<host_view_type>();
		const local_index_type field_dim_i = field_dims[i];
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,local_coords_size), KOKKOS_LAMBDA(const int j) {
			for (local_index_type k=0; k<field_dim_i; k++) host_data(j,k) = field_values[i][j*field_dim_i + k];
		});

		field->syncMemory();
	}

	if (_keep_original_lat_lon)	{
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
			lat_host_data(j,0) = lat_vals[j];
			lon_host_data(j,0) = lon_vals[j];
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




#endif // COMPADRE_USE_NETCDF_MPI
#endif // COMPADRE_USE_NETCDF

#ifdef COMPADRE_USE_VTK

int LegacyVTKFileIO::read(const std::string& fn) {
	vtkSmartPointer<vtkPolyDataReader> reader =
		vtkSmartPointer<vtkPolyDataReader>::New();
	reader->SetFileName(fn.c_str());
	reader->ReadAllScalarsOn();
	reader->ReadAllVectorsOn();
	reader->Register(reader);
	reader->Update();
	_dataSet = vtkDataSet::SafeDownCast(reader->GetOutput());

	coords_type* coords = _particles->getCoords();

	nPtsGlobal = _dataSet->GetNumberOfPoints();

//			std::cout << "initializing " << nPtsGlobal << " coordinates...\n";
//						std::cout << "initializing map ...\n";

	_particles->resize(nPtsGlobal);

	// Now check for point data
	vtkPointData *pd = _dataSet->GetPointData();
	if (pd)
	{
//				std::cout << nPtsGlobal << " points contains point data with "
//					<< pd->GetNumberOfArrays() << " "
//					<< pd->GetNumberOfComponents()
//					<< " arrays." << std::endl;

		// global indices but are local to this processor
		minInd = coords->getMinGlobalIndex();
		maxInd = coords->getMaxGlobalIndex();

				std::cout << "min: " << minInd << " max: " << maxInd << std::endl;
				std::cout << "compare len: " << coords->nLocal() << " to " << maxInd-minInd << std::endl;

		// first fill the coordinates
		host_view_type dev_coords = coords->getPts()->getLocalView<host_view_type>(); // assumes we are reading in physical coords in a lagrangian simulation

		const local_index_type ndim = coords->nDim();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int i) {
			double coord[ndim];
			_dataSet->GetPoint(i, coord);
			for (local_index_type j=0; j<coords->nDim(); j++) dev_coords(i-minInd,j) = coord[j];
		});

		// sync coords and fields after the fill
//		coords->syncMemory();
		//coords->print(std::cout);

		for (int i = 0; i < pd->GetNumberOfArrays(); i++) {
			_particles->getCoordsConst()->getComm()->barrier();
		// 	std::cout << "\tArray " << i
		// 	<< " is named "
		// 	<< (pd->GetArrayName(i) ? pd->GetArrayName(i) : "NULL")
		// 	<< " with "
		// 	<< std::endl;

			const vtkIdType comp_size = pd->GetArray(i)->GetNumberOfComponents();
			std::string lowerCaseArrayName = pd->GetArrayName(i);
			transform(lowerCaseArrayName.begin(), lowerCaseArrayName.end(), lowerCaseArrayName.begin(), ::tolower);
			if ( std::strcmp(lowerCaseArrayName.c_str(), "bc_id") == 0  || std::strcmp(lowerCaseArrayName.c_str(), "flag") == 0 ) {
				host_view_type bc_id = _particles->getFlags()->getLocalView<host_view_type>();
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int j) {
					double data[comp_size];
					pd->GetArray(i)->GetTuple(j, data);
					bc_id(j-minInd,0) = data[0];
				});
			} else {
				Teuchos::RCP<field_type> field = _particles->getFieldManager()->createField(
						comp_size, (pd->GetArrayName(i) ? pd->GetArrayName(i) : "NULL"), "null");

				// fill portion of vector corresponding to global ids that are located on this processor
				host_view_type dev_data = field->getLocalVectorVals()->getLocalView<host_view_type>();
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(minInd,maxInd+1), KOKKOS_LAMBDA(const int j) {
					double data[comp_size];
					pd->GetArray(i)->GetTuple(j, data);
					for (local_index_type k=0; k<comp_size; k++) dev_data(j-minInd,k) = data[k];
				});
				field->syncMemory();
			}
		 }
	}

	if (this->_particles->getCoords()->isLagrangian()) this->_particles->snapLagrangianCoordsToPhysicalCoords();
	return 1;
}


int XMLVTUFileIO::read(const std::string& fn) {
	local_index_type comm_size = _particles->getCoordsConst()->getComm()->getSize();
	local_index_type my_rank = _particles->getCoordsConst()->getComm()->getRank();

	MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
	vtkMPICommunicatorOpaqueComm mpi_comm(&comm);
	vtkSmartPointer<vtkMPICommunicator> communicator = vtkSmartPointer<vtkMPICommunicator>::New();
	communicator->InitializeExternal(&mpi_comm);
	vtkSmartPointer<vtkMPIController> controller = vtkSmartPointer<vtkMPIController>::New();
	controller->SetCommunicator(communicator);

	vtkSmartPointer<vtkXMLPUnstructuredGridReader> parallel_reader =
		vtkSmartPointer<vtkXMLPUnstructuredGridReader>::New();
	parallel_reader->SetFileName(fn.c_str());
	parallel_reader->UpdateInformation();

	vtkInformation* outInfo = parallel_reader->GetOutputInformation(0);
	outInfo->Set(vtkAlgorithm::CAN_HANDLE_PIECE_REQUEST(), 1);
	outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_PIECE_NUMBER(), controller->GetLocalProcessId());
	outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_PIECES(), controller->GetNumberOfProcesses());
	outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_GHOST_LEVELS(), 1);
	parallel_reader->Update();

	vtkSmartPointer<vtkVertexGlyphFilter> arrows = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	// this filter deletes all other cell data
	arrows->SetInputData(parallel_reader->GetOutput());
	arrows->Update();

	vtkSmartPointer<vtkAppendFilter> preAppendFilter =
	    vtkSmartPointer<vtkAppendFilter>::New();
	preAppendFilter->AddInputData(arrows->GetOutput());
	preAppendFilter->AddInputData(parallel_reader->GetOutput());
	preAppendFilter->Update();

	vtkSmartPointer<vtkDistributedDataFilter> dd =
		vtkSmartPointer<vtkDistributedDataFilter>::New();
	dd->SetController(controller);
	dd->SetInputData(preAppendFilter->GetOutput());
	dd->SetBoundaryModeToAssignToAllIntersectingRegions();
	dd->UseMinimalMemoryOff(); // less communication, more memory use
//	dd->GetCuts();
	dd->SetRetainKdtree(1);
	dd->Update();

	_dataSet =
		vtkSmartPointer<vtkUnstructuredGrid>::New();

	_dataSet->DeepCopy(dd->GetOutput());

	this->populateParticles();

	controller->Finalize(1);

	if (this->_particles->getCoords()->isLagrangian()) this->_particles->snapLagrangianCoordsToPhysicalCoords();
	return 1;
}


int XMLVTPFileIO::read(const std::string& fn) {
	local_index_type comm_size = _particles->getCoordsConst()->getComm()->getSize();
	local_index_type my_rank = _particles->getCoordsConst()->getComm()->getRank();

	MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
	vtkMPICommunicatorOpaqueComm mpi_comm(&comm);
	vtkSmartPointer<vtkMPICommunicator> communicator = vtkSmartPointer<vtkMPICommunicator>::New();
	communicator->InitializeExternal(&mpi_comm);
	vtkSmartPointer<vtkMPIController> controller = vtkSmartPointer<vtkMPIController>::New();
	controller->SetCommunicator(communicator);

	vtkSmartPointer<vtkXMLPPolyDataReader> parallel_reader =
		vtkSmartPointer<vtkXMLPPolyDataReader>::New();
	parallel_reader->SetFileName(fn.c_str());
	parallel_reader->UpdateInformation();

	vtkInformation* outInfo = parallel_reader->GetOutputInformation(0);
	outInfo->Set(vtkAlgorithm::CAN_HANDLE_PIECE_REQUEST(), 1);
	outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_PIECE_NUMBER(), controller->GetLocalProcessId());
	outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_PIECES(), controller->GetNumberOfProcesses());
	outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_GHOST_LEVELS(), 1);
	parallel_reader->Update();

	vtkSmartPointer<vtkVertexGlyphFilter> arrows = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	// this filter deletes all other cell data
	arrows->SetInputData(parallel_reader->GetOutput());
	arrows->Update();

	vtkSmartPointer<vtkAppendFilter> preAppendFilter =
	    vtkSmartPointer<vtkAppendFilter>::New();
	preAppendFilter->AddInputData(arrows->GetOutput());
	preAppendFilter->AddInputData(parallel_reader->GetOutput());
	preAppendFilter->Update();

	vtkSmartPointer<vtkDistributedDataFilter> dd =
		vtkSmartPointer<vtkDistributedDataFilter>::New();
	dd->SetController(controller);
	dd->SetInputData(preAppendFilter->GetOutput());
	dd->SetBoundaryModeToAssignToAllIntersectingRegions();
	dd->UseMinimalMemoryOff(); // less communication, more memory use
//	dd->GetCuts();
	dd->SetRetainKdtree(1);
	dd->Update();

	_dataSet =
		vtkSmartPointer<vtkUnstructuredGrid>::New();

	_dataSet->DeepCopy(dd->GetOutput());
	std::cout<<"to populate" << std::endl;
	this->populateParticles();std::cout<<"past populate" << std::endl;

	controller->Finalize(1);

	if (this->_particles->getCoords()->isLagrangian()) this->_particles->snapLagrangianCoordsToPhysicalCoords();
	return 1;
}

#endif // COMPADRE_USE_VTK

#ifdef COMPADRE_USE_NETCDF

void SerialNetCDFFileIO::write(const std::string& fn, bool use_binary) {
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
	const local_index_type coords_size = host_coords.dimension_0();
	const local_index_type ndim = coords->nDim();

	// define the dimensions
	if ((retval = nc_def_dim(ncid, "particle", coords_size, &particle_dim_id)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval!=0, "Error.");

	// define the spatial coordinate variables
	retval = nc_def_var(ncid, "x", NC_DOUBLE, 1, &particle_dim_id, &x_var_id);
	retval = nc_def_var(ncid, "y", NC_DOUBLE, 1, &particle_dim_id, &y_var_id);
	retval = nc_def_var(ncid, "z", NC_DOUBLE, 1, &particle_dim_id, &z_var_id);
	std::string units = this->_particles->getParameters()->get<Teuchos::ParameterList>("coordinates").get<std::string>("units");
	retval = nc_put_att_text(ncid, x_var_id, "units", units.length(), units.c_str());
	retval = nc_put_att_text(ncid, y_var_id, "units", units.length(), units.c_str());
	retval = nc_put_att_text(ncid, z_var_id, "units", units.length(), units.c_str());

	// loop over fields and define for each of them
	const std::vector<Teuchos::RCP<field_type> > fields = _particles->getFieldManagerConst()->getVectorOfFields();
	std::vector<local_index_type> field_dim_ids(fields.size());
	std::vector<local_index_type> field_var_ids(fields.size());
	for (local_index_type i=0; i<fields.size(); i++){
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
	retval = nc_def_var(ncid, "FLAG", NC_INT, 1, &particle_dim_id, &bc_var_id);
	retval = nc_put_att_text(ncid, bc_var_id, "units", 4, "none");

	local_index_type ids_var_id;
	retval = nc_def_var(ncid, "ID", NC_INT64, 1, &particle_dim_id, &ids_var_id);
	retval = nc_put_att_text(ncid, ids_var_id, "units", 4, "none");

	// end of defining dimensions and variables
	retval = nc_enddef(ncid);


	std::vector<scalar_type> coords_x(coords_size);
	std::vector<scalar_type> coords_y(coords_size);
	std::vector<scalar_type> coords_z(coords_size);
	for (int i=0; i<coords_size; i++) {
		coords_x[i] = host_coords(i,0);
		coords_y[i] = host_coords(i,1);
		coords_z[i] = host_coords(i,2);
	}

	retval = nc_put_var(ncid, x_var_id, &coords_x[0]);
	retval = nc_put_var(ncid, y_var_id, &coords_y[0]);
	retval = nc_put_var(ncid, z_var_id, &coords_z[0]);

	for (local_index_type i=0; i<fields.size(); i++){

		std::vector<scalar_type> host_field_data_vec(coords_size*fields[i]->nDim());

		host_view_type host_field_data = fields[i]->getLocalVectorVals()->getLocalView<host_view_type>();
//		{
//			// ONLY FOR TESTING!!!!!
//			for (int j=0; j<3; j++) {
//				for (int k=0; k<coords_size; k++) {
//					host_field_data(k,j) = j;
//				}
//			}
//		}
		const local_index_type field_i_ndim = fields[i]->nDim();
		for (int j=0; j<coords_size; j++) {
			for (int k=0; k<field_i_ndim; k++) {
				host_field_data_vec[field_i_ndim*j+k] = host_field_data(j,k);
			}
		}
		retval = nc_put_var(ncid, field_var_ids[i], &host_field_data_vec[0]);
	}


	{
		host_view_type bc_id = _particles->getFlags()->getLocalView<host_view_type>();
		std::vector<global_index_type> bc_id_vec(coords_size);
		for (int k=0; k<coords_size; k++) {
			bc_id_vec[k] = bc_id(k,0);
		}
		retval = nc_put_var(ncid, bc_var_id, &bc_id_vec[0]);

		typedef Kokkos::View<const global_index_type*> const_gid_view_type;
		const_gid_view_type gids = _particles->getCoordsConst()->getMapConst()->getMyGlobalIndices();

		std::vector<global_index_type> gids_vec(coords_size);
		for (int k=0; k<coords_size; k++) {
			gids_vec[k] = gids(k,0);
		}
		retval = nc_put_var(ncid, ids_var_id, &gids_vec[0]);
	}

	// Close the file
	if ((retval = nc_close(ncid)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not closed successfully.");
}

#ifdef COMPADRE_USE_NETCDF_MPI

void ParallelHDF5NetCDFFileIO::write(const std::string& fn, bool use_binary) {
	#define NC_INDEPENDENT 0
	#define NC_COLLECTIVE 1

	MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
	MPI_Info info = MPI_INFO_NULL;

	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int ncid, retval;
	if (( retval = nc_create_par (      fn.c_str(),
										NC_NETCDF4|NC_MPIIO,
										comm,
										info,
										&ncid
										)))

		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not created successfully.");

//	std::cout << "rank " << rank << " of " << size << std::endl;

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
	const local_index_type coords_size = host_coords.dimension_0();
	const local_index_type ndim = coords->nDim();

	// define the dimensions
	if ((retval = nc_def_dim(ncid, "particle", global_coords_size, &particle_dim_id)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval!=0, "Error.");

	// define the spatial coordinate variables
	retval = nc_def_var(ncid, "x", NC_DOUBLE, 1, &particle_dim_id, &x_var_id);
	retval = nc_def_var(ncid, "y", NC_DOUBLE, 1, &particle_dim_id, &y_var_id);
	retval = nc_def_var(ncid, "z", NC_DOUBLE, 1, &particle_dim_id, &z_var_id);
	std::string units = this->_particles->getParameters()->get<Teuchos::ParameterList>("coordinates").get<std::string>("units");
	retval = nc_put_att_text(ncid, x_var_id, "units", units.length(), units.c_str());
	retval = nc_put_att_text(ncid, y_var_id, "units", units.length(), units.c_str());
	retval = nc_put_att_text(ncid, z_var_id, "units", units.length(), units.c_str());

	// loop over fields and define for each of them
	const std::vector<Teuchos::RCP<field_type> > fields = _particles->getFieldManagerConst()->getVectorOfFields();
	std::vector<local_index_type> field_dim_ids(fields.size());
	std::vector<local_index_type> field_var_ids(fields.size());
	for (local_index_type i=0; i<fields.size(); i++){
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
	retval = nc_def_var(ncid, "FLAG", NC_INT, 1, &particle_dim_id, &bc_var_id);
	retval = nc_put_att_text(ncid, bc_var_id, "units", 4, "none");

	local_index_type ids_var_id;
	retval = nc_def_var(ncid, "ID", NC_INT64, 1, &particle_dim_id, &ids_var_id);
	if ((retval = nc_put_att_text(ncid, ids_var_id, "units", 4, "none")))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "id units didn't finish.");

	// end of defining dimensions and variables
	if ((retval = nc_enddef(ncid)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "definitions didn't finish.");

//	std::cout << coords_size << " on " << rank << std::endl;
	std::vector<scalar_type> coords_x(coords_size);
	std::vector<scalar_type> coords_y(coords_size);
	std::vector<scalar_type> coords_z(coords_size);
	for (int i=0; i<coords_size; i++) {
		coords_x[i] = host_coords(i,0);
		coords_y[i] = host_coords(i,1);
		coords_z[i] = host_coords(i,2);
	}

	if ((retval = nc_var_par_access(ncid, x_var_id, NC_COLLECTIVE)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "x write permission not changed.");
	if ((retval = nc_var_par_access(ncid, y_var_id, NC_COLLECTIVE)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "y write permission not changed.");
	if ((retval = nc_var_par_access(ncid, z_var_id, NC_COLLECTIVE)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "z write permission not changed.");

	// this mapping temporarily gives the min and max elements which is equivalent to an offset
	Teuchos::RCP<map_type> temporary_map = Teuchos::rcp(new map_type(
		Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), coords_size, 0, _particles->getCoordsConst()->getComm()));

	{
		unsigned long start = (unsigned long)(temporary_map->getMinGlobalIndex());
		unsigned long countDiff = (unsigned long)(coords_size);

		if ((retval = nc_put_vara(ncid, x_var_id, &start, &countDiff, &coords_x[0])))
			TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "x not written.");
		if ((retval = nc_put_vara(ncid, y_var_id, &start, &countDiff, &coords_y[0])))
			TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "y not written.");
		if ((retval = nc_put_vara(ncid, z_var_id, &start, &countDiff, &coords_z[0])))
			TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "z not written.");
	}


	for (local_index_type i=0; i<fields.size(); i++){
		std::vector<scalar_type> host_field_data_vec(coords_size*fields[i]->nDim());

		host_view_type host_field_data = fields[i]->getLocalVectorVals()->getLocalView<host_view_type>();
//		{
//			// ONLY FOR TESTING!!!!!
//			for (int j=0; j<3; j++) {
//				for (int k=0; k<coords_size; k++) {
//					host_field_data(k,j) = j;
//				}
//			}
//		}
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

		host_view_type bc_id = _particles->getFlags()->getLocalView<host_view_type>();
		std::vector<global_index_type> bc_id_vec(coords_size);
		for (int k=0; k<coords_size; k++) {
			bc_id_vec[k] = bc_id(k,0);
		}
		retval = nc_put_vara(ncid, bc_var_id, &start, &countDiff, &bc_id_vec[0]);

		typedef Kokkos::View<const global_index_type*> const_gid_view_type;
		const_gid_view_type gids = _particles->getCoordsConst()->getMapConst()->getMyGlobalIndices();

		std::vector<global_index_type> gids_vec(coords_size);
		for (int k=0; k<coords_size; k++) {
			gids_vec[k] = gids(k,0);
		}
		retval = nc_put_vara(ncid, ids_var_id, &start, &countDiff, &gids_vec[0]);
	}

	// Close the file
	if ((retval = nc_close(ncid)))
		TEUCHOS_TEST_FOR_EXCEPT_MSG(retval, "File not closed successfully.");
}

#endif // COMPADRE_USE_NETCDF
#endif // COMPADRE_USE_NETCDF_MPI

#ifdef COMPADRE_USE_VTK

void LegacyVTKFileIO::write(const std::string& fn, bool use_binary) {
	local_index_type comm_size = _particles->getCoordsConst()->getComm()->getSize();
	local_index_type my_rank = _particles->getCoordsConst()->getComm()->getRank();

	MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
	vtkMPICommunicatorOpaqueComm mpi_comm(&comm);
	vtkSmartPointer<vtkMPICommunicator> communicator = vtkSmartPointer<vtkMPICommunicator>::New();
	communicator->InitializeExternal(&mpi_comm);
	vtkSmartPointer<vtkMPIController> controller = vtkSmartPointer<vtkMPIController>::New();
	controller->SetCommunicator(communicator);

	Teuchos::RCP<VTKData> vtk_data = Teuchos::rcp(new Compadre::VTKData(_particles,_particles->getParameters()));
	vtk_data->generateDataSet();
	_dataSet = vtk_data->getDataSet();


	vtkSmartPointer<vtkPDataSetWriter> parallel_writer =
			vtkSmartPointer<vtkPDataSetWriter>::New();

	parallel_writer->SetFileName(fn.c_str());
	parallel_writer->SetInputData(_dataSet);

	parallel_writer->SetNumberOfPieces(comm_size); // hard coded to same as comm_size for now
	parallel_writer->SetStartPiece(my_rank);
	parallel_writer->SetEndPiece(my_rank);
	parallel_writer->SetGhostLevel(0);
	parallel_writer->SetController(controller);
	parallel_writer->Write();

	controller->Finalize(1);
}


void XMLVTUFileIO::write(const std::string& fn, bool use_binary) {
	local_index_type comm_size = _particles->getCoordsConst()->getComm()->getSize();
	local_index_type my_rank = _particles->getCoordsConst()->getComm()->getRank();

	MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
	vtkMPICommunicatorOpaqueComm mpi_comm(&comm);
	vtkSmartPointer<vtkMPICommunicator> communicator = vtkSmartPointer<vtkMPICommunicator>::New();
	communicator->InitializeExternal(&mpi_comm);
	vtkSmartPointer<vtkMPIController> controller = vtkSmartPointer<vtkMPIController>::New();
	controller->SetCommunicator(communicator);


	if (!_unstructuredGrid) {
		Teuchos::RCP<VTKData> vtk_data = Teuchos::rcp(new Compadre::VTKData(_particles,_particles->getParameters()));
		vtk_data->generateDataSet();
		_dataSet = vtk_data->getDataSet();

		vtkSmartPointer<vtkAppendFilter> appendFilter =
			vtkSmartPointer<vtkAppendFilter>::New();
		appendFilter->AddInputData(_dataSet);
		appendFilter->Update();

		_unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
		_unstructuredGrid->DeepCopy(appendFilter->GetOutput());
	}


	vtkSmartPointer<vtkXMLPUnstructuredGridWriter> parallel_writer =
			vtkSmartPointer<vtkXMLPUnstructuredGridWriter>::New();


	parallel_writer->SetFileName(fn.c_str());


	parallel_writer->SetInputData(_unstructuredGrid);
	if (use_binary)
		parallel_writer->SetDataModeToAppended();
	else parallel_writer->SetDataModeToAscii();

	parallel_writer->SetNumberOfPieces(comm_size); // hard coded to same as comm_size for now
	parallel_writer->SetStartPiece(my_rank);
	parallel_writer->SetEndPiece(my_rank);
	parallel_writer->SetController(controller);
	parallel_writer->Update();

	controller->Finalize(1);
}


void XMLVTPFileIO::write(const std::string& fn, bool use_binary) {
	local_index_type comm_size = _particles->getCoordsConst()->getComm()->getSize();
	local_index_type my_rank = _particles->getCoordsConst()->getComm()->getRank();

	MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
	vtkMPICommunicatorOpaqueComm mpi_comm(&comm);
	vtkSmartPointer<vtkMPICommunicator> communicator = vtkSmartPointer<vtkMPICommunicator>::New();
	communicator->InitializeExternal(&mpi_comm);
	vtkSmartPointer<vtkMPIController> controller = vtkSmartPointer<vtkMPIController>::New();
	controller->SetCommunicator(communicator);

	if (!_polyData) { // polyData could have been created be generateMesh()
		Teuchos::RCP<VTKData> vtk_data = Teuchos::rcp(new Compadre::VTKData(_particles, _particles->getParameters()));
		vtk_data->generateDataSet();
		_dataSet = vtk_data->getDataSet();

		_polyData = vtkPolyData::SafeDownCast(_dataSet);
	}

	vtkSmartPointer<vtkXMLPPolyDataWriter> parallel_writer =
			vtkSmartPointer<vtkXMLPPolyDataWriter>::New();

	parallel_writer->SetFileName(fn.c_str());
	parallel_writer->SetInputData(_polyData);

	if (use_binary)
		parallel_writer->SetDataModeToAppended();
	else parallel_writer->SetDataModeToAscii();

	parallel_writer->SetNumberOfPieces(comm_size); // hard coded to same as comm_size for now
	parallel_writer->SetStartPiece(my_rank);
	parallel_writer->SetEndPiece(my_rank);
	parallel_writer->SetController(controller);
	parallel_writer->Update();

	controller->Finalize(1);
}

void XMLVTUFileIO::generateMesh() {
	/*
	 *  There remains a problem with overlapping vtu ghost points where whichever
	 *  processor is read in last overwrites when viewed in Paraview
	 *  ( currently only a problem for bc_id/flag variables because all other fields have
	 *   the correct values on the halo particles )
	 */

	{
		Teuchos::RCP<VTKData> vtk_data = Teuchos::rcp(new Compadre::VTKData(_particles, _particles->getParameters()));
		vtk_data->generateDataSet(true /*halo*/);
		_dataSet = vtk_data->getDataSet();
		_haloDataSet = vtk_data->getHaloDataSet();

		vtkSmartPointer<vtkAppendFilter> appendFilter =
			vtkSmartPointer<vtkAppendFilter>::New();
		appendFilter->AddInputData(_dataSet);
		appendFilter->Update();

		_unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
		_unstructuredGrid->DeepCopy(appendFilter->GetOutput());
	}

//	{
//		_dataSet->AllocatePointGhostArray();
//		_haloDataSet->AllocatePointGhostArray();
//		vtkUnsignedCharArray* ghosts = _haloDataSet->GetPointGhostArray();
//		for (int i=0; i<_haloDataSet->GetNumberOfPoints(); i++) {
//			ghosts->SetValue(i, ghosts->GetValue(i) | vtkDataSetAttributes::HIDDENPOINT + vtkDataSetAttributes::DUPLICATEPOINT);
//		}
//	}

//	{
//		_dataSet->AllocatePointGhostArray();
//		_haloDataSet->AllocatePointGhostArray();
//		vtkUnsignedCharArray* ghost_vals = _haloDataSet->GetPointGhostArray();
//		const local_index_type ghost_num = ghost_vals->GetNumberOfTuples();
//
//		Kokkos::parallel_for(Kokkos::RangePolicy<>(0,ghost_num), KOKKOS_LAMBDA(const int j) {
//			double data[1];
//			data[0]=1;
//			ghost_vals->SetTuple(j,data);
//		});
//	}

	vtkSmartPointer<vtkAppendFilter> appendFilter =
		vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter->AddInputData(_dataSet);
	appendFilter->AddInputData(_haloDataSet);
	appendFilter->Update();

//	vtkSmartPointer<vtkVertexGlyphFilter> arrows = vtkSmartPointer<vtkVertexGlyphFilter>::New();
//	// this filter deletes all other cell data
//	arrows->SetInputData(appendFilter->GetOutput());
//	arrows->Update();
//
//	vtkSmartPointer<vtkAppendFilter> secondAppendFilter =
//	    vtkSmartPointer<vtkAppendFilter>::New();
//	secondAppendFilter->AddInputData(arrows->GetOutput());
//	secondAppendFilter->AddInputData(appendFilter->GetOutput());
//	secondAppendFilter->Update();
//
//	vtkSmartPointer<vtkDistributedDataFilter> dd =
//		vtkSmartPointer<vtkDistributedDataFilter>::New();
//	vtkInformation* outInfo = dd->GetOutputInformation(0);
//	outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_NUMBER_OF_GHOST_LEVELS(), 1);
//	dd->SetController(controller);
//	dd->SetInputData(secondAppendFilter->GetOutput());
//	dd->SetBoundaryModeToAssignToAllIntersectingRegions();
////	dd->SetBoundaryModeToSplitBoundaryCells();//ToAssignToAllIntersectingRegions();
//	dd->UseMinimalMemoryOff(); // less communication, more memory use
//	dd->Update();

	vtkSmartPointer<vtkDelaunay3D> delaunay3D =
	    vtkSmartPointer<vtkDelaunay3D>::New();
	delaunay3D->SetInputData(appendFilter->GetOutput());
	delaunay3D->Update();

	_mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
	_mesh->DeepCopy(delaunay3D->GetOutput());

//	controller->Finalize(1);
}

void XMLVTPFileIO::generateMesh() {
	/*
	 *  There remains a problem with overlapping vtu ghost points where whichever
	 *  processor is read in last overwrites when viewed in Paraview
	 *  ( currently only a problem for bc_id/flag variables because all other fields have
	 *   the correct values on the halo particles )
	 */
	{
		Teuchos::RCP<VTKData> vtk_data = Teuchos::rcp(new Compadre::VTKData(_particles, _particles->getParameters()));
		vtk_data->generateDataSet(true /*halo*/);
		_dataSet = vtk_data->getDataSet();
		_haloDataSet = vtk_data->getHaloDataSet();

		_polyData = vtkPolyData::SafeDownCast(_dataSet);
	}

	vtkSmartPointer<vtkAppendFilter> appendFilter =
		vtkSmartPointer<vtkAppendFilter>::New();
	appendFilter->AddInputData(_dataSet);
	appendFilter->AddInputData(_haloDataSet);
	appendFilter->Update();

	vtkSmartPointer<vtkDelaunay3D> delaunay3D =
	    vtkSmartPointer<vtkDelaunay3D>::New();
	delaunay3D->SetInputData(appendFilter->GetOutput());
	delaunay3D->Update();

	_mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
	_mesh->DeepCopy(delaunay3D->GetOutput());
}

void VTKFileIO::writeMesh(const std::string& fn) {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(!_mesh, "writeMesh() called before generateMesh()");
	local_index_type comm_size = _particles->getCoordsConst()->getComm()->getSize();
	local_index_type my_rank = _particles->getCoordsConst()->getComm()->getRank();

	MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
	vtkMPICommunicatorOpaqueComm mpi_comm(&comm);
	vtkSmartPointer<vtkMPICommunicator> communicator = vtkSmartPointer<vtkMPICommunicator>::New();
	communicator->InitializeExternal(&mpi_comm);
	vtkSmartPointer<vtkMPIController> controller = vtkSmartPointer<vtkMPIController>::New();
	controller->SetCommunicator(communicator);


	vtkSmartPointer<vtkXMLPUnstructuredGridWriter> ugWriter =
		  vtkSmartPointer<vtkXMLPUnstructuredGridWriter>::New();
	ugWriter->SetInputData(_mesh);
	ugWriter->SetFileName(fn.c_str());

	ugWriter->SetNumberOfPieces(comm_size); // hard coded to same as comm_size for now
	ugWriter->SetStartPiece(my_rank);
	ugWriter->SetEndPiece(my_rank);
	ugWriter->SetController(controller);
	ugWriter->Update();

	controller->Finalize(1);
}

#endif // COMPADRE_USE_VTK

}
