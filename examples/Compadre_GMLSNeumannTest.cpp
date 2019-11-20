#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_nanoflannInformation.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <Compadre_GMLS.hpp>

// #include <Compadre_GMLS_PinnedLaplacianNeumann_Sources.hpp>
// #include <Compadre_GMLS_PinnedLaplacianNeumann_BoundaryConditions.hpp>
#include <Compadre_GMLS_PinnedLaplacianNeumann_Operator.hpp>

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::XyzVector xyz_type;
typedef Compadre::EuclideanCoordsT CT;

int main (int argc, char* args[]) {
#ifdef TRILINOS_LINEAR_SOLVES
    Teuchos::RCP<Compadre::ParameterManager> parameter_manager;
    if (argc > 1) {
        parameter_manager = Teuchos::rcp(new Compadre::ParameterManager(argc, args));
    } else {
        parameter_manager = Teuchos::rcp(new Compadre::ParameterManager());
        std::cout << "WARNING: No parameter list given. Default parameters used." << std::endl;
    }
    if (parameter_manager->helpRequested()) return 0;
    if (parameter_manager->parseError()) return -1;

    Teuchos::RCP<Teuchos::ParameterList> parameters = parameter_manager->getList();

    Teuchos::GlobalMPISession mpi(&argc, &args);
    Teuchos::oblackholestream bstream;
    Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::DefaultComm<int>::getComm();

    Kokkos::initialize(argc, args);

    Teuchos::RCP<Teuchos::Time> FirstReadTime = Teuchos::TimeMonitor::getNewCounter("1st Read Time");
    Teuchos::RCP<Teuchos::Time> AssemblyTime = Teuchos::TimeMonitor::getNewCounter("Assembly Time");
    Teuchos::RCP<Teuchos::Time> SolvingTime = Teuchos::TimeMonitor::getNewCounter("Solving Time");
    Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter("Write Time");
    Teuchos::RCP<Teuchos::Time> SecondReadTime = Teuchos::TimeMonitor::getNewCounter("2nd Read Time");

    // This proceeds setting up the problem so that the parameters will be propagated down into the physics and bcs
    try {
        parameters->get<std::string>("solution type");
    } catch (Teuchos::Exceptions::InvalidParameter & exception) {
        parameters->set<std::string>("solution type", "sine");
    }

    {
        std::vector<std::string> fnames(3);
        std::vector<double> hsize(3);
        std::vector<double> errors(3);
        const std::string filename_prefix = parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix");
        fnames[0] = filename_prefix + "6.nc";
        fnames[1] = filename_prefix + "12.nc";
        fnames[2] = filename_prefix + "24.nc";
        hsize[0] = 6;
        hsize[1] = 12;
        hsize[2] = 24;

        TEUCHOS_TEST_FOR_EXCEPT_MSG(parameters->get<int>("loop size")>3, "Only three mesh levels available for this problem.");

        for (LO i=0; i<parameters->get<int>("loop size"); i++) {
            int Porder = parameters->get<Teuchos::ParameterList>("remap").get<int>("porder");
            double h_size = 2.0/hsize[i];

            std::string testfilename(fnames[i]);

            Teuchos::RCP<Compadre::ParticlesT> particles = Teuchos::rcp( new Compadre::ParticlesT(parameters, comm) );
            const CT* coords = (CT*)(particles->getCoordsConst());

            // Read in data file.
            FirstReadTime->start();
            Compadre::FileManager fm;
            fm.setReader(testfilename, particles);
            fm.read();
            FirstReadTime->stop();

            particles->zoltan2Initialize();
            particles->getFieldManager()->printAllFields(std::cout);

            ST halo_size;
            {
                if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
                    halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
                } else {
                    halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
                }
                particles->buildHalo(halo_size);
                particles->createDOFManager();

                //Set the radius for the neighbor list:
                ST h_support;
                if (parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("dynamic radius")) {
                    h_support = h_size;
                } else {
                    h_support = parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size");
                }
                particles->createNeighborhood();
                particles->getNeighborhood()->setAllHSupportSizes(h_support);

                LO neighbors_needed = Compadre::GMLS::getNP(Porder);

                LO extra_neighbors = parameters->get<Teuchos::ParameterList>("remap").get<double>("neighbors needed multiplier") * neighbors_needed;
                particles->getNeighborhood()->constructAllNeighborList(particles->getCoordsConst()->getHaloSize(), extra_neighbors);

                // Iterative solver for the problem
                Teuchos::RCP<Compadre::ProblemT> problem = Teuchos::rcp( new Compadre::ProblemT(particles));

                // construct physics, sources, and boundary conditions
                Teuchos::RCP<Compadre::GMLS_PinnedLaplacianNeumannPhysics> physics =
                    Teuchos::rcp( new Compadre::GMLS_PinnedLaplacianNeumannPhysics(particles, Porder));

                // set physics, sources and boundary conditions in the problem
                problem->setPhysics(physics);
            }
        }
    }

#endif
    return 0;
}
