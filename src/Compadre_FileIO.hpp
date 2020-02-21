#ifndef _COMPADRE_FILEIO_
#define _COMPADRE_FILEIO_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

class ParticlesT;


// utility function for getting extension of file
std::string getFilenameExtension(const std::string& filename);

enum CoordinateLayout {XYZ_separate, XYZ_joint, LAT_LON_separate};

class FileIO {
	/*
		Base class for all FileIO
	*/
	protected:

		typedef Compadre::ParticlesT particles_type;

		particles_type* _particles;

        CoordinateLayout _coordinate_layout;
        std::vector<std::string> _coordinate_names;
        std::string _particle_num_name;

        bool _keep_original_coordinates;
        std::string _lat_lon_unit_type;
        std::string _coordinate_unit_name;

		virtual void populateParticles() = 0;

	public:

		FileIO ( particles_type * particles ) : _particles(particles), _coordinate_layout(XYZ_separate) {
            _coordinate_names.resize(3,"");
            _coordinate_names[0]="x";
            _coordinate_names[1]="y";
            _coordinate_names[2]="z";
            _particle_num_name="num_entities";

            _keep_original_coordinates = false; // keeps lat-lon when converting to xyz, e.g.
		}
		virtual ~FileIO() {};

		virtual int read(const std::string& fn) = 0;
		virtual void write(const std::string& fn, bool use_binary) = 0;

        // coordinates will be stored in a rank-2 tensor with one axis being the number of particles [particle_num_name]
        // and the other axis being the number of pieces of coordinate data per particle
        // in XYZ_joint, there is one variable name, where particle_num_name is used for the length of the array
        // in XYZ_separate, there are multiple variable names (3), where particle_num_name is used for the length of each array, and 1 is the width of each
        // in LAT_LON_separate, there are multiple variable names (2), where particle_num_name is used for the length of each array, and 1 is the width of each
        void setCoordinateLayout(std::string particle_num_name, std::string layout_name, std::string c0 = "x", std::string c1 = "y", std::string c2 = "z") {
            _particle_num_name=particle_num_name;
	        transform(layout_name.begin(), layout_name.end(), layout_name.begin(), ::tolower);
            if (layout_name == "xyz_separate") {
                _coordinate_layout = XYZ_separate;
                _coordinate_names[0]=c0;
                _coordinate_names[1]=c1;
                _coordinate_names[2]=c2;
            } else if (layout_name == "xyz_joint") {
                _coordinate_layout = XYZ_joint;
                _coordinate_names[0]=c0;
            } else if (layout_name == "lat_lon_separate") {
                _coordinate_layout = LAT_LON_separate;
                _coordinate_names[0]=c0;
                _coordinate_names[1]=c1;
            } else TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Invalid layout_name provided to setCoordinateLayout()");
        }

        void setCoordinateUnitStyle(std::string units) { 
	        transform(units.begin(), units.end(), units.begin(), ::tolower);
            if (_coordinate_layout == LAT_LON_separate) {
                if (units=="radians" || units=="radian") _lat_lon_unit_type = "radians";
                else if (units=="degrees" || units=="degree") _lat_lon_unit_type = "degrees";
                else if (units=="none") _lat_lon_unit_type = "none";
                else TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Invalid unit type provided to setLatLonUnitStyle");
            } // other coordinate layouts unaffected  by unit style
        }

        void setCoordinateUnitName(std::string units) { 
            // not currently used as only fields have units
            //_coordinate_unit_name = units;
        }

        void setKeepOriginalCoordinates(bool keep) { _keep_original_coordinates = keep; }

        void copySettingsFrom(const FileIO &another_fileio) {
		    _particles = another_fileio._particles;

            _coordinate_layout = another_fileio._coordinate_layout;
            _coordinate_names = another_fileio._coordinate_names;
            _particle_num_name = another_fileio._particle_num_name;

            _keep_original_coordinates = another_fileio._keep_original_coordinates;
            _lat_lon_unit_type = another_fileio._lat_lon_unit_type;
            _coordinate_unit_name = another_fileio._coordinate_unit_name;
        }
};

#ifdef COMPADREHARNESS_USE_NETCDF

class NetCDFFileIO : public FileIO {
	/*
		NOTE: NetCDF Specialization of FileIO base class
	*/
	protected:

		virtual void populateParticles() = 0;

	public:

		NetCDFFileIO ( particles_type * particles ) : FileIO(particles) {}
		virtual ~NetCDFFileIO() {};

		virtual int read(const std::string& fn) = 0;
		virtual void write(const std::string& fn, bool use_binary) = 0;

};

class SerialNetCDFFileIO : public NetCDFFileIO {
	/*
		NOTE: Parallel NetCDF Specialization of FileIO base class
	*/
	protected:

		virtual void populateParticles(){};

	public:

		SerialNetCDFFileIO ( particles_type * particles ) : NetCDFFileIO(particles) {}
		virtual ~SerialNetCDFFileIO() {};

		virtual int read(const std::string& fn);
		virtual void write(const std::string& fn, bool use_binary);

};


#ifdef COMPADREHARNESS_USE_NETCDF_MPI

class ParallelHDF5NetCDFFileIO : public NetCDFFileIO {
	/*
		NOTE: Parallel NetCDF Specialization of FileIO base class
	*/
	protected:

		virtual void populateParticles(){};

	public:

		ParallelHDF5NetCDFFileIO ( particles_type * particles ) : NetCDFFileIO(particles) {}
		virtual ~ParallelHDF5NetCDFFileIO() {};

		virtual int read(const std::string& fn);
		virtual void write(const std::string& fn, bool use_binary);

};


#endif // COMPADREHARNESS_USE_NETCDF_MPI
#endif // COMPADREHARNESS_USE_NETCDF


class FileManager {

	protected:

		typedef Compadre::ParticlesT particles_type;

		Teuchos::RCP<Compadre::FileIO> _io;
		std::string _reader_fn;
		std::string _writer_fn;
		bool _use_binary;
		std::string _type;
		particles_type* _particles;

        void readerIOChoice(std::string& extension, const bool enforce_serial = false);
        void writerIOChoice(std::string& extension, const bool enforce_serial = false);

        bool _is_reader;
        bool _is_writer;

	public:

		FileManager() : _reader_fn("\0"), _writer_fn("\0"), _use_binary(true), _particles(NULL), 
                        _is_reader(false), _is_writer(false) {};

		~FileManager() {};

		void setReader(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const std::string& type = std::string());

        Compadre::FileIO* getReader() { return _io.getRawPtr(); }

		void setWriter(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const bool use_binary = true);

		void read();

		void write() const;

};

}

#endif
