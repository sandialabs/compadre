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

class FileIO {
	/*
		Base class for all FileIO
	*/
	protected:

		typedef Compadre::ParticlesT particles_type;

		particles_type* _particles;

		global_index_type nPtsGlobal;
		global_index_type minInd;
		global_index_type maxInd;
		local_index_type nPtsLocal;

		virtual void populateParticles() = 0;

	public:

		FileIO ( particles_type * particles ) : _particles(particles) {
			nPtsGlobal=0;
			minInd=0;
			maxInd=0;
			nPtsLocal=0;
		}
		virtual ~FileIO() {};

		virtual int read(const std::string& fn) = 0;
		virtual void write(const std::string& fn, bool use_binary) = 0;

		virtual void generateMesh() = 0;
		virtual void writeMesh(const std::string& fn) = 0;

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

class SerialHOMMEFileIO : public SerialNetCDFFileIO {
	/*
		NOTE: Parallel NetCDF Specialization of FileIO base class
	*/
	protected:

		virtual void populateParticles(){};
		const std::string& _lat_lon_units;
		const bool _keep_original_lat_lon;

	public:

		SerialHOMMEFileIO ( particles_type * particles, const bool keep_original_lat_lon = false, const std::string& lat_lon_units = std::string("radians") ) :
			SerialNetCDFFileIO(particles), _lat_lon_units(lat_lon_units), _keep_original_lat_lon(keep_original_lat_lon)  {}
		virtual ~SerialHOMMEFileIO() {};

		virtual int read(const std::string& fn);
//		virtual void write(const std::string& fn, bool use_binary);

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

class ParallelHOMMEFileIO : public ParallelHDF5NetCDFFileIO {
	/*
		NOTE: Parallel NetCDF Specialization of FileIO base class
	*/
	protected:

		virtual void populateParticles(){};
		const std::string& _lat_lon_units;
		const bool _keep_original_lat_lon;

	public:

		ParallelHOMMEFileIO ( particles_type * particles, const bool keep_original_lat_lon = false, const std::string& lat_lon_units = std::string("radians") ) :
			ParallelHDF5NetCDFFileIO(particles), _lat_lon_units(lat_lon_units), _keep_original_lat_lon(keep_original_lat_lon) {}

		virtual ~ParallelHOMMEFileIO() {};

		virtual int read(const std::string& fn);
//		virtual void write(const std::string& fn, bool use_binary);

		virtual void generateMesh(){};
		virtual void writeMesh(const std::string& fn){};

};

class ParallelMPASFileIO : public NetCDFFileIO {
	/*
		NOTE: Parallel NetCDF Specialization of FileIO base class
	*/
	protected:

		virtual void populateParticles(){};

	public:

		ParallelMPASFileIO ( particles_type * particles ) : NetCDFFileIO(particles) {}
		virtual ~ParallelMPASFileIO() {};

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
		bool _is_parallel;
		std::string _type;
		particles_type* _particles;

	public:

		FileManager() : _use_binary(true), _is_parallel(false), _particles(NULL) {};
		~FileManager() {};

		void setReader(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const std::string& type = std::string(), const bool keep_original_lat_lon = false);

		void setWriter(const std::string& _fn, Teuchos::RCP<ParticlesT>& particles, const bool use_binary = true);

		void read() const;

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
