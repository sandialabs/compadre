#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include "Compadre_ParticlesT.hpp"
#include <Compadre_FieldManager.hpp>
#include "Compadre_SphericalCoordsT.hpp"
#include "Compadre_EuclideanCoordsT.hpp"
#include "Compadre_FieldT.hpp"
#include "Compadre_AnalyticFunctions.hpp"
#include "Compadre_FileIO.hpp"
#include "Compadre_ParameterManager.hpp"
#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;

void writeToCSV(Compadre::SphericalCoordsT& ec, Compadre::FieldT& f, std::ostream& os);

int main (int argc, char* args[]) {

	Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::oblackholestream bstream;
	Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

	std::string output_file_prefix("../test_data/");
	
	Kokkos::initialize(argc, args);
//	{
//		const int procRank = comm->getRank();
//		const int nProcs = comm->getSize();
//// 		std::cout << "Hello from MPI rank " << procRank << ", Kokkos execution space "
//// 				  << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
//
//		std::string testfilename("./build/icosTri_1.nc");
//
//
//		Compadre::ASCIIFileReader<Compadre::SphericalCoordsT<ST, LO, GO>, ST, LO, GO> reader(testfilename);
//	// 	GO npts = reader.getNGlobalCoordsFromFile(comm);
//	// 	std::cout << "found " << npts << " coordinates in file " << testfilename << std::endl;
//		Teuchos::RCP<Compadre::SphericalCoordsT<ST, LO, GO> > sphCoords = Teuchos::rcp( new Compadre::SphericalCoordsT<ST, LO, GO>(0, comm));
//		Teuchos::RCP<Compadre::ParticlesT> null_particles =
//				Teuchos::rcp( new Compadre::ParticlesT(comm, Teuchos::rcp_dynamic_cast<Compadre::CoordsT<ST,LO,GO> >(sphCoords)));
//		reader.initCoordsFromSingleFile(null_particles, comm, 4);
//
//		std::cout << *sphCoords << std::endl;
//
//		{
//			std::stringstream ss;
//			ss << "sphCoordsBefore" << procRank << ".m";
//			std::ofstream file(ss.str());
//			sphCoords->writeToMatlab(file, "sphImport", procRank);
//			file.close();
//		}
//		sphCoords->zoltan2Init();
//		sphCoords->zoltan2Partition();
//
//		{
//			std::stringstream ss;
//			ss << "sphCoordsAfter" << procRank << ".m";
//			std::ofstream file(ss.str());
//			sphCoords->writeToMatlab(file, "sphImport", procRank);
//			file.close();
//		}
//	}
	{
		const int procRank = comm->getRank();

		std::string testfilename(args[1]);

		Teuchos::RCP<Compadre::ParameterManager> parameter_manager = Teuchos::rcp(new Compadre::ParameterManager());
		Teuchos::RCP<Teuchos::ParameterList> parameters = parameter_manager->getList();
		Teuchos::ParameterList coordList = parameters->get<Teuchos::ParameterList>("coordinates");
		coordList.set("type","spherical");

		typedef Compadre::SphericalCoordsT CT;

		Teuchos::RCP<CT> sphCoords = Teuchos::rcp( new CT(0, comm));
		Teuchos::RCP<Compadre::ParticlesT> particles =
				Teuchos::rcp( new Compadre::ParticlesT(parameters, Teuchos::rcp_dynamic_cast<Compadre::CoordsT>(sphCoords)));

		Compadre::FileManager fm;
		fm.setReader(testfilename, particles);
		fm.read();

		std::cout << *sphCoords << std::endl;

		{
			std::stringstream ss;
			ss << output_file_prefix << "sphCoordsBefore" << procRank << ".m";
			std::ofstream file(ss.str());
			sphCoords->writeToMatlab(file, "sphImport", procRank);
			file.close();
		}
//		sphCoords->zoltan2Init();
//		sphCoords->zoltan2Partition();

		{
			std::stringstream ss;
			ss << output_file_prefix << "sphCoordsAfter" << procRank << ".m";
			std::ofstream file(ss.str());
			sphCoords->writeToMatlab(file, "sphImport", procRank);
			file.close();
		}

		particles->getFieldManagerConst()->printAllFields(std::cout);

		std::string output_testfilename(output_file_prefix + "out.nc");
		fm.setWriter(output_testfilename, particles);
		fm.write();
	}

	// NETCDF reader test
	{
		std::string testfilename("out.nc");

		Teuchos::RCP<Compadre::ParameterManager> parameter_manager = Teuchos::rcp(new Compadre::ParameterManager());
		Teuchos::RCP<Teuchos::ParameterList> parameters = parameter_manager->getList();
		Teuchos::ParameterList coordList = parameters->get<Teuchos::ParameterList>("coordinates");
		coordList.set("type","euclidean");

		typedef Compadre::EuclideanCoordsT CT;

		Teuchos::RCP<CT> sphCoords = Teuchos::rcp( new CT(0, comm));
		Teuchos::RCP<Compadre::ParticlesT> particles =
				Teuchos::rcp( new Compadre::ParticlesT(parameters, Teuchos::rcp_dynamic_cast<Compadre::CoordsT>(sphCoords)));

		Compadre::FileManager fm;
		fm.setReader(output_file_prefix + testfilename, particles);
		fm.read();

		std::cout << *sphCoords << std::endl;

		particles->getFieldManagerConst()->printAllFields(std::cout);
	}
	Kokkos::finalize();
	return 0;
}
