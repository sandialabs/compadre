#ifndef _COMPADRE_FILEIO_
#define _COMPADRE_FILEIO_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

#ifdef COMPADREHARNESS_USE_VTK
	#include <vtkSmartPointer.h>
#endif

class vtkDataSet;
class vtkPolyData;
class vtkUnstructuredGrid;

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

		virtual void generateMesh() = 0;
		virtual void writeMesh(const std::string& fn) = 0;

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

		virtual void generateMesh() = 0;
		virtual void writeMesh(const std::string& fn) = 0;

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

		virtual void generateMesh(){};
		virtual void writeMesh(const std::string& fn){};

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

		virtual void generateMesh(){};
		virtual void writeMesh(const std::string& fn){};

};


#endif // COMPADREHARNESS_USE_NETCDF_MPI
#endif // COMPADREHARNESS_USE_NETCDF

#ifdef COMPADREHARNESS_USE_VTK

class VTKFileIO : public FileIO {
	/*
		NOTE: VTK Specialization of FileIO base class
	*/
	protected:

		vtkSmartPointer<vtkDataSet> _dataSet;
		vtkSmartPointer<vtkDataSet> _haloDataSet;
		vtkSmartPointer<vtkUnstructuredGrid> _mesh; // doesn't exist unless explicitly created

		virtual void populateParticles();

	public:

		VTKFileIO ( particles_type * particles ) : FileIO(particles) {}
		virtual ~VTKFileIO() {};

		virtual int read(const std::string& fn) = 0;
		virtual void write(const std::string& fn, bool use_binary) = 0;

		virtual void generateMesh() = 0;
		virtual void writeMesh(const std::string& fn);

};

class LegacyVTKFileIO : public VTKFileIO {
	/*
		NOTE: Serial reader / writer for legacy VTK (.vtk) files
	*/

	public:

		LegacyVTKFileIO ( particles_type * particles ) : VTKFileIO(particles) {}
		virtual ~LegacyVTKFileIO() {};

		virtual int read(const std::string& fn);
		virtual void write(const std::string& fn, bool use_binary = false);

		virtual void generateMesh() {};

};

class XMLVTUFileIO : public VTKFileIO {
	/*
		NOTE: XML Parallel UnstructuredGrid reader / writer for VTK (format is .pvtu with supporting .vtu files)
	*/
	protected:
		vtkSmartPointer<vtkUnstructuredGrid> _unstructuredGrid;

	public:

		XMLVTUFileIO ( particles_type * particles ) : VTKFileIO(particles) {}
		virtual ~XMLVTUFileIO() {};

		virtual int read(const std::string& fn);
		virtual void write(const std::string& fn, bool use_binary = true);

		virtual void generateMesh();

};

class XMLVTPFileIO : public VTKFileIO {
	/*
		NOTE: XML Parallel PolyData reader / writer for VTK (format is .pvtp with supporting .vtp files)
	*/
	protected:

		vtkSmartPointer<vtkPolyData> _polyData;

	public:

		XMLVTPFileIO ( particles_type * particles ) : VTKFileIO(particles) {}
		virtual ~XMLVTPFileIO() {};

		virtual int read(const std::string& fn);
		virtual void write(const std::string& fn, bool use_binary = true);

		virtual void generateMesh();

};

#endif // COMPADREHARNESS_USE_VTK

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

		FileManager() : _use_binary(true), _particles(NULL), _is_reader(false), _is_writer(false) {};
		~FileManager() {};

		void setReader(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const std::string& type = std::string());

        Compadre::FileIO* getReader() { return _io.getRawPtr(); }

		void setWriter(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const bool use_binary = true);

		void read();

		void write() const;

		void generateWriteMesh();

};

#ifdef COMPADREHARNESS_USE_VTK

class VTKData {
	/*
	 * Not for files. This generates a dataset given a set of particles. It can be used by FileIO as well as in neighbor searches.
	 */
	protected:

		typedef Compadre::ParticlesT particles_type;

		const particles_type* _particles;
		vtkSmartPointer<vtkDataSet> _dataSet;
		vtkSmartPointer<vtkDataSet> _haloDataSet;
		vtkSmartPointer<vtkDataSet> _data_w_halo_DataSet;
		Teuchos::RCP<Teuchos::ParameterList> _parameters;

	public:

		VTKData ( const particles_type * particles , Teuchos::RCP<Teuchos::ParameterList> parameters) :
			_particles(particles), _parameters(parameters) {}
		~VTKData() {};

		vtkSmartPointer<vtkDataSet> getDataSet() const { return _dataSet; }
		vtkSmartPointer<vtkDataSet> getHaloDataSet() const { return _haloDataSet; }
		vtkSmartPointer<vtkDataSet> getCombinedDataSet() const {
			TEUCHOS_TEST_FOR_EXCEPT_MSG(_data_w_halo_DataSet==NULL, "getCombinedDataSet() called before generateCombinedDataSet()");
			return _data_w_halo_DataSet;
		}

		void generateDataSet( bool include_halo = false, bool for_writing_output = true, bool use_physical_coords = true);
		void generateCombinedDataSet();
};

#endif // COMPADREHARNESS_USE_VTK
}

#endif
