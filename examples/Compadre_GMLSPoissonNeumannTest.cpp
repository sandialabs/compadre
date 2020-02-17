#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <Compadre_GMLS.hpp>

#include <Compadre_GMLS_PoissonNeumann_Operator.hpp>
#include <Compadre_GMLS_PoissonNeumann_BoundaryConditions.hpp>
#include <Compadre_GMLS_PoissonNeumann_Sources.hpp>

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
            particles->getFieldManager()->createField(1, "neumann solution");
            // particles->getFieldManager()->printAllFields(std::cout);

            ST halo_size;
            {
                if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
                    halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
                } else {
                    halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
                }
                particles->buildHalo(halo_size);
                particles->createDOFManager();

                particles->createNeighborhood();

                LO neighbors_needed = Compadre::GMLS::getNP(Porder);

 		        particles->getNeighborhood()->constructAllNeighborLists(particles->getCoordsConst()->getHaloSize(),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
                    true /*dry run for sizes*/,
                    neighbors_needed,
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));

                // Iterative solver for the problem
                Teuchos::RCP<Compadre::ProblemT> problem = Teuchos::rcp( new Compadre::ProblemT(particles));

                // construct physics, sources, and boundary conditions
                Teuchos::RCP<Compadre::GMLS_PoissonNeumannPhysics> physics =
                    Teuchos::rcp( new Compadre::GMLS_PoissonNeumannPhysics(particles, Porder));
                Teuchos::RCP<Compadre::GMLS_PoissonNeumannSources> source =
                    Teuchos::rcp( new Compadre::GMLS_PoissonNeumannSources(particles));
                Teuchos::RCP<Compadre::GMLS_PoissonNeumannBoundaryConditions> bcs =
                    Teuchos::rcp( new Compadre::GMLS_PoissonNeumannBoundaryConditions(particles));

                // set physics, sources and boundary conditions in the problem
                problem->setPhysics(physics);
                source->setPhysics(physics);

                problem->setSources(source);
                problem->setBCS(bcs);

                // assembly
                AssemblyTime->start();
                problem->initialize();
                AssemblyTime->stop();

                // solving
                SolvingTime->start();
                problem->solve();
                SolvingTime->stop();
            }

            // check solution
            double norm = 0.0;
            double exact = 0.0;

            Teuchos::RCP<Compadre::AnalyticFunction> function;
            if (parameters->get<std::string>("solution type")=="sine") {
                function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts));
            } else {
                function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SecondOrderBasis));
            }
            // store exact solutions onto particles' field
            particles->getFieldManager()->createField(1, "exact solution");
            auto function1 = Compadre::SineProducts();
            particles->getFieldManager()->getFieldByName("exact solution")->localInitFromScalarFunction(&function1);

            for (int j=0; j<coords->nLocal(); j++) {
                xyz_type xyz = coords->getLocalCoords(j);
                const ST val = particles->getFieldManagerConst()->getFieldByName("neumann solution")->getLocalScalarVal(j);
                exact = function->evalScalar(xyz);
                norm += (exact - val)*(exact - val);
            }
            norm /= (double)(coords->nGlobalMax());

            ST global_norm;
            Teuchos::Ptr<ST> global_norm_ptr(&global_norm);
            Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, norm, global_norm_ptr);
            global_norm = sqrt(global_norm);
            if (comm->getRank()==0) std::cout << "Global Norm: " << global_norm << "\n";
            errors[i] = global_norm;

            WriteTime->start();
            std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + std::to_string(i) /* loop */ + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
            std::string writetest_output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "writetest" + std::to_string(i) /* loop */ + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
            fm.setWriter(output_filename, particles);
            fm.write();
            WriteTime->stop();

            // if (parameters->get<Teuchos::ParameterList>("solver").get<std::string>("type")=="direct")
            //     Teuchos::TimeMonitor::summarize();

            TEUCHOS_TEST_FOR_EXCEPT_MSG(errors[i]!=errors[i], "NaN found in error norm.");
            if (parameters->get<std::string>("solution type")=="sine") {
                if (i>0)
                    TEUCHOS_TEST_FOR_EXCEPT_MSG(errors[i-1]/errors[i] < 3.0, std::string("Second order not achieved for sine solution (should be 3). Is: ") + std::to_string(errors[i-1]/errors[i]));
            } else {
                TEUCHOS_TEST_FOR_EXCEPT_MSG(errors[i] > 1e-13, "Second order solution not recovered exactly.");
            }
        }
        // if (comm->getRank()==0) parameters->print();
    }
    // Teuchos::TimeMonitor::summarize();
    Kokkos::finalize();
#endif
    return 0;
}
