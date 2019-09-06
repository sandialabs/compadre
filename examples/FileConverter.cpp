#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>

#include <Kokkos_Core.hpp>

#include <Compadre_GMLS.hpp> // for getNP()
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_SphericalCoordsT.hpp>
#include <Compadre_nanoflannInformation.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include "Compadre_GlobalConstants.hpp"
static const Compadre::GlobalConstants consts;

#include <iostream>

#define STACK_TRACE(call) try { call; } catch (const std::exception& e ) { TEUCHOS_TRACE(e); }

using namespace Compadre;

int main (int argc, char* args[]) {

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
    Teuchos::GlobalMPISession mpi(&argc, &args);
    Teuchos::oblackholestream bstream;
    Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

    Kokkos::initialize(argc, args);
    
    const int procRank = comm->getRank();
    const int nProcs = comm->getSize();

    {

        // SOURCE PARTICLES
        typedef Compadre::EuclideanCoordsT CT;
        Teuchos::RCP<Compadre::ParticlesT> particles =
                Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
         CT* coords = (CT*)particles->getCoords();

        //READ IN SOURCE DATA FILE
        {
            Compadre::FileManager fm;
            std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));
            fm.setReader(testfilename, particles);
            fm.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
            fm.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
            fm.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
            fm.read();
        }

        // WRITE TO NEW FILE HERE
        {
            Compadre::FileManager fm3;
            std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
            fm3.setWriter(output_filename, particles);
            fm3.write();
        }
    }

    if (comm->getRank()==0) parameters->print();
    Kokkos::finalize();
    return 0;
}


