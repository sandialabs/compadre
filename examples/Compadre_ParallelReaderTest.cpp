#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;

int main (int argc, char* args[]) {
	//********* MPI & KOKKOS SETUP
	Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::oblackholestream bstream;
	Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
	Kokkos::initialize(argc, args);
	//*********

	//********* TIMERS
	Teuchos::RCP<Teuchos::Time> ParameterTime = Teuchos::TimeMonitor::getNewCounter ("Parameter Initialization");
	Teuchos::RCP<Teuchos::Time> ParticleGenerationTime = Teuchos::TimeMonitor::getNewCounter ("Initialize Particles");
	Teuchos::RCP<Teuchos::Time> ReadTime = Teuchos::TimeMonitor::getNewCounter ("Read Time");
	Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");
	//*********

	//********* BASIC PARAMETER SETUP FOR ANY PROBLEM
	ParameterTime->start();
	Teuchos::RCP<Compadre::ParameterManager> parameter_manager;
	if (argc > 1)
		parameter_manager = Teuchos::rcp(new Compadre::ParameterManager(argc, args));
	else {
		parameter_manager = Teuchos::rcp(new Compadre::ParameterManager());
		std::cout << "WARNING: No parameter list given. Default parameters used." << std::endl;
	}
	if (parameter_manager->helpRequested()) return 0;
	if (parameter_manager->parseError()) return -1;

	Teuchos::RCP<Teuchos::ParameterList> parameters = parameter_manager->getList();
	parameters->print();
	ParameterTime->stop();
	//*********

	{
		std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));

		typedef Compadre::EuclideanCoordsT CT;

		ParticleGenerationTime->start();
		Teuchos::RCP<CT> coords = Teuchos::rcp( new CT(0, comm));
		Teuchos::RCP<Compadre::ParticlesT> particles =
				Teuchos::rcp( new Compadre::ParticlesT(parameters, Teuchos::rcp_dynamic_cast<Compadre::CoordsT>(coords)));
		ParticleGenerationTime->stop();

		// try to read in the file which was based on either 2 or 4 processors
		Compadre::FileManager fm;
		ReadTime->start();
		fm.setReader(testfilename, particles);
		fm.read();
		ReadTime->stop();
		particles->getFieldManagerConst()->printAllFields(std::cout);

		// just to verify what was actually read in
		std::cout << parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file") << " file" << std::endl;
		WriteTime->start();
		fm.setWriter(parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file"), particles);
		fm.write();
		WriteTime->stop();
	}

	Teuchos::TimeMonitor::summarize();
	comm->barrier();
	Kokkos::finalize();
	return 0;
}
