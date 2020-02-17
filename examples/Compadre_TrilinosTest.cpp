#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_SphericalCoordsT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <Compadre_PinnedGraphLaplacian_Operator.hpp>
#include <Compadre_PinnedGraphLaplacian_Sources.hpp>
#include <Compadre_PinnedGraphLaplacian_BoundaryConditions.hpp>

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;

struct KokkosHello{
    const int rank;
    
    KokkosHello(const int _rank) : rank(_rank) {};
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const int i) const {
        std::printf("rank %d, i = %d\n", rank, i);
    }
};

void writeToCSV(Compadre::EuclideanCoordsT& ec, Compadre::FieldT& f, std::ostream& os);

int main (int argc, char* args[]) {
#ifdef TRILINOS_LINEAR_SOLVES

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

    Teuchos::RCP<Teuchos::Time> SphericalParticleTime = Teuchos::TimeMonitor::getNewCounter ("Spherical Particle Time");
    Teuchos::RCP<Teuchos::Time> NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Trilinos Test - Neighbor Search Time");
    Teuchos::RCP<Teuchos::Time> FirstReadTime = Teuchos::TimeMonitor::getNewCounter ("1st Read Time");
    Teuchos::RCP<Teuchos::Time> AssemblyTime = Teuchos::TimeMonitor::getNewCounter ("Assembly Time");
    Teuchos::RCP<Teuchos::Time> SolvingTime = Teuchos::TimeMonitor::getNewCounter ("Solving Time");
    Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");
    Teuchos::RCP<Teuchos::Time> SecondReadTime = Teuchos::TimeMonitor::getNewCounter ("2nd Read Time");

    std::cout << "Hello from MPI rank " << procRank << ", Kokkos execution space "
              << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Kokkos::parallel_for(20, KokkosHello(procRank));

    {
        /*
         *  Tests Euclidean particles by DIRECTLY constructing, partitioning, building halos, and applying evaluation of a scalar on the coordinates
         */
        const LO n = 20;
        const GO n3 = n * n * n;

        typedef Compadre::EuclideanCoordsT CT;
        Teuchos::RCP<CT> eucCoords = Teuchos::rcp(new CT(n3, comm));
        eucCoords->initRandom(2.0, procRank);

        std::stringstream ss;
        ss << parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") << "eucMvec" << procRank << ".m";
        std::ofstream file(ss.str());
        eucCoords->writeToMatlab(file, "eucBefore", procRank);

        eucCoords->zoltan2Init();
        eucCoords->zoltan2Partition();
        eucCoords->writeToMatlab(file, "eucAfterMJ", procRank);
        file.close();

        {
            // generate and update halo
            eucCoords->buildHalo(0.1);
            std::stringstream halo_ss;
            halo_ss << parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") << "eucMvecHalo" << procRank << ".m";
            std::ofstream halo_file(halo_ss.str());
            eucCoords->writeHaloToMatlab(halo_file, "eucHaloMJ", procRank);
            halo_file.close();
        }

        Compadre::FieldT gaussField(eucCoords.getRawPtr(), 1, "gauss");
        Compadre::Gaussian3D fn;
        Compadre::AnalyticFunction* fnPtr = &fn;
        gaussField.localInitFromScalarFunction(fnPtr);

        std::vector<LO> indices_to_remove;
        indices_to_remove.push_back(0);
        indices_to_remove.push_back(3);
        indices_to_remove.push_back(7);
        indices_to_remove.push_back(1999);
        eucCoords->removeCoords(indices_to_remove);

        std::vector<Compadre::XyzVector> verts_to_insert;
        verts_to_insert.push_back(Compadre::XyzVector(1,0,0));
        verts_to_insert.push_back(Compadre::XyzVector(1,1,0));
        verts_to_insert.push_back(Compadre::XyzVector(1,1,1));
        verts_to_insert.push_back(Compadre::XyzVector(0,0,0));
        eucCoords->insertCoords(verts_to_insert);

        ss.str("");
        ss << parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") << "eucData" << procRank << ".csv";
        std::ofstream csvFile(ss.str());
        writeToCSV(*eucCoords, gaussField, csvFile);
        csvFile.close();
    }
    {
        /*
         *  Tests Spherical particles by DIRECTLY constructing but then partitioning, building halos, and solving through Particles
         */
        const LO nlat = 91;
        const LO nlon = 180;
        const GO npts = nlat * nlon;

        Compadre::GlobalConstants consts;
        const ST h_size = 2.0*consts.Pi()/((double)nlat);

        typedef Compadre::SphericalCoordsT CT;

        Teuchos::RCP<CT> sphCoords = Teuchos::rcp(new CT(npts, comm));
        sphCoords->initRandom(1.0, procRank);

        std::stringstream ss;
        ss << parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") << "sphMvec" << procRank << ".m";
        std::ofstream file(ss.str());
        sphCoords->writeToMatlab(file, "sphBefore", procRank);

        // Zoltan2 partitioning performed directly on coordinates prior to instantiation of particles
        // This tests that particles can correctly derive flag, field sizes, etc... with a preexisting partition
        sphCoords->zoltan2Init();
        sphCoords->zoltan2Partition();

        sphCoords->writeToMatlab(file, "sphAfterMJ", procRank);

        file.close();

        Teuchos::ParameterList coordList = parameters->get<Teuchos::ParameterList>("coordinates");
        coordList.set("type","spherical");

         Teuchos::RCP<Compadre::ParticlesT> particles =
             Teuchos::rcp( new Compadre::ParticlesT(parameters, Teuchos::rcp_dynamic_cast<Compadre::CoordsT>(sphCoords)));

         ST halo_size;
         {
             if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
                 halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
             } else {
                 halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
             }
             particles->buildHalo(halo_size);
            std::stringstream halo_ss;
            halo_ss << parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") << "sphMvecHalo" << procRank << ".m";
            std::ofstream halo_file(halo_ss.str());
            sphCoords->writeHaloToMatlab(halo_file, "sphHaloMJ", procRank);
            halo_file.close();
         }

         // this tests adding fields after generating the partition and halo information
         particles->getFieldManager()->createField(3, "velocity", "cm/s");
         particles->getFieldManager()->createField(1, "pressure", "Pa");
         particles->getFieldManager()->createField(1, "density", "kg/cm^3");
         particles->getFieldManager()->createField(1, "placeholder1", "kg/cm^3");
         particles->getFieldManager()->createField(2, "placeholder2", "kg/cm^3");
         particles->getFieldManager()->listFields(std::cout);
         particles->createDOFManager();

         LO pressure_field_id = particles->getFieldManagerConst()->getIDOfFieldFromName("pressure");
         TEUCHOS_TEST_FOR_EXCEPT_MSG(pressure_field_id != 1, "Returned id of " + std::to_string(pressure_field_id) + ", but should have returned 1.");

         LO velocity_field_id = particles->getFieldManagerConst()->getIDOfFieldFromName("velocity");
                  TEUCHOS_TEST_FOR_EXCEPT_MSG(velocity_field_id != 0, "Returned id of " + std::to_string(velocity_field_id) + ", but should have returned 0.");

         const ST h_support = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
         particles->createNeighborhood();
         particles->getNeighborhood()->setAllHSupportSizes(h_support);
         particles->getNeighborhood()->constructAllNeighborLists(particles->getCoordsConst()->getHaloSize(),
            parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
            true /*dry run for sizes*/,
            7 /* neighbors needed */,
            parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
            parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
            parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
            parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));

         Teuchos::RCP<Compadre::ProblemT> problem =
             Teuchos::rcp( new Compadre::ProblemT(particles) );

         Teuchos::RCP<Compadre::PinnedGraphLaplacianPhysics> physics = Teuchos::rcp( new Compadre::PinnedGraphLaplacianPhysics(particles));
         Teuchos::RCP<Compadre::PinnedGraphLaplacianSources> sources = Teuchos::rcp( new Compadre::PinnedGraphLaplacianSources(particles));
         Teuchos::RCP<Compadre::PinnedGraphLaplacianBoundaryConditions> bcs = Teuchos::rcp( new Compadre::PinnedGraphLaplacianBoundaryConditions(particles));

         problem->setBCS(bcs);
         problem->setPhysics(physics);
         problem->setSources(sources);

         Teuchos::RCP<Compadre::BoundaryConditionsT> bc = problem->getBCS();
         bc->flagBoundaries();
         problem->initialize();
         problem->solve();

         // currently this test goes through the motions of a solver, but really it has not source terms so the result is zero

         // Should get warning Warning! Solver has experienced a loss of accuracy! because no point is pinned
        std::cout << "global N from reduce all = " << sphCoords->nGlobal() << std::endl;
    }
    {

        {
            //
            // VTK File Reader Test and Parallel VTK File Reader
            //


            FirstReadTime->start();
            std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));

            typedef Compadre::SphericalCoordsT CT;

            Teuchos::RCP<Compadre::ParticlesT> particles =
                    Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));

            Compadre::FileManager fm;
            fm.setReader(testfilename, particles);
            fm.read();
//            particles->getFieldManager()->printAllFields(std::cout);
            //particles->getCoords()->print(std::cout);
            FirstReadTime->stop();
            ST halo_size;

            particles->getFieldManagerConst()->listFields(std::cout);

            {
                const ST h_size = .0833333;

                particles->zoltan2Initialize(); // also applies the repartition to coords, flags, and fields
                 if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
                     halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
                 } else {
                     halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
                 }
                 particles->buildHalo(halo_size);
                 particles->createDOFManager();

                NeighborSearchTime->start();
                 const ST h_support = parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size");

                 particles->createNeighborhood();
                particles->getNeighborhood()->setAllHSupportSizes(h_support);
                 particles->getNeighborhood()->constructAllNeighborLists(particles->getCoordsConst()->getHaloSize(),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
                    true /*dry run for sizes*/,
                    4 /* neighbors needed */,
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));
                NeighborSearchTime->stop();

                Teuchos::RCP<Compadre::ProblemT> problem =
                    Teuchos::rcp( new Compadre::ProblemT(particles));

                // construct physics, sources, and boundary conditions
                Teuchos::RCP<Compadre::PinnedGraphLaplacianPhysics> physics =
                    Teuchos::rcp( new Compadre::PinnedGraphLaplacianPhysics(particles));
                Teuchos::RCP<Compadre::PinnedGraphLaplacianSources> source =
                    Teuchos::rcp( new Compadre::PinnedGraphLaplacianSources(particles));
                Teuchos::RCP<Compadre::PinnedGraphLaplacianBoundaryConditions> bcs =
                    Teuchos::rcp( new Compadre::PinnedGraphLaplacianBoundaryConditions(particles));

                // set physics, sources, and boundary conditions in the problem
                problem->setPhysics(physics);
                problem->setSources(source);
                problem->setBCS(bcs);

                // assembly
                AssemblyTime->start();
                problem->initialize();
                AssemblyTime->stop();

                //solving
                SolvingTime->start();
                problem->solve();
                SolvingTime->stop();
            }

            // set each flag to the global id for particle at i
            auto gids = particles->getCoordsConst()->getMapConst()->getMyGlobalIndices();
            const CT * sphCoords = (CT*)(particles->getCoordsConst());
            for (int i=0; i<sphCoords->nLocal(); i++) {
                particles->setFlag(i,(int)(gids(i)));
            }

            WriteTime->start();
            {
                std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "temp_" + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
                Compadre::FileManager fm;
                fm.setWriter(output_filename, particles);
                fm.write();
            }
            WriteTime->stop();

            // removal of particles destroys map, doesn't preserve gids on reconstructions
            //std::vector<LO> indices_to_remove;
            //for (LO j=50; j<201; j++) indices_to_remove.push_back(j);
            //particles->removeParticles(indices_to_remove);

            WriteTime->start();
            {
                std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
                Compadre::FileManager fm;
                fm.setWriter(output_filename, particles, false /* don't use binary */);
                fm.write();
            }
            WriteTime->stop();
        }
        {
             //
             // NetCDF File Reader Test and Parallel NetCDF File Re-Reader
             //

             // read solution back in which should also reset flags

             SecondReadTime->start();

             typedef Compadre::SphericalCoordsT CT;

            // set parameters to preserve ID
            parameters->get<Teuchos::ParameterList>("io").set<bool>("preserve gids", true);

             Teuchos::RCP<Compadre::ParticlesT> particles =
                 Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
             const CT * sphCoords = (CT*)(particles->getCoordsConst());

            Compadre::FileManager fm;
            fm.setReader(parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file"), particles);
            fm.read();
             SecondReadTime->stop();

             particles->getFieldManagerConst()->listFields(std::cout);

             // since preserve gids is true, value of flag at particle i should match 
             // global id of particle at i 
             // this tests setting gids as flags, writing flags and storing gids,
             // reading back in preserved gid ordering (determining flag values)
             // which are then read back in and checked against the original
            auto gids = particles->getCoordsConst()->getMapConst()->getMyGlobalIndices();
             for (int i=0; i<sphCoords->nLocal(); i++) {
                 std::ostringstream msg;
                 msg << "bc_id read in wrong value for " << gids(i) << " of " << particles->getFlag(i) << " instead of " << gids(i);
                 // accomodate for removing particles 50 through 200 from each processor
                 TEUCHOS_TEST_FOR_EXCEPT_MSG((int)(gids(i))-particles->getFlag(i) != 0, msg.str());
            }
            // preserve gids must be set back to false since it is incompatible with insertParticles
            parameters->get<Teuchos::ParameterList>("io").set<bool>("preserve gids", false);
            comm->barrier();

             // needed for insertParticles to perform an interpolation
             particles->zoltan2Initialize();
             particles->buildHalo(1.0);
             particles->createNeighborhood();

             Compadre::SecondOrderBasis second_order_sol;
             Compadre::SineProducts sinx_sol;
             particles->getFieldManager()->getFieldByName("P")->localInitFromScalarFunction(&second_order_sol);
             particles->getFieldManager()->getFieldByName("F")->localInitFromVectorFunction(&second_order_sol);
             particles->getFieldManager()->createField(3,"sinx","m^2");
             particles->getFieldManager()->getFieldByName("sinx")->localInitFromVectorFunction(&sinx_sol);
             particles->getFieldManager()->updateFieldsHaloData();
             // vector u is left intact from solution, rho is left intact as a scalar from solution

            std::vector<Compadre::XyzVector> verts_to_insert;
            auto proc_mins = particles->getCoords()->boundingBoxMinOnProcessor(comm->getRank());
            auto proc_maxs = particles->getCoords()->boundingBoxMinOnProcessor(comm->getRank());
            std::vector<Compadre::XyzVector> points = {
                Compadre::XyzVector(.9,0,0),
                Compadre::XyzVector(.9,.9,0),
                Compadre::XyzVector(.9,.9,.9),
                Compadre::XyzVector(.1,.1,.1),
                Compadre::XyzVector(0.75,.9,0.5),
                Compadre::XyzVector(0.0,.75,0.5)
            };

            for (auto point=points.begin(); point!=points.end(); ++point) {
                if (particles->getCoords()->verifyCoordsOnProcessor(std::vector<Compadre::XyzVector>(1,*point))) {
                    verts_to_insert.push_back(*point);
                    printf("Point added: (%f,%f,%f) on %d\n", (*point)[0], (*point)[1], (*point)[2], comm->getRank());
                }
            }

            particles->insertParticles(verts_to_insert);

             // Temporary, writer out additional particle data fields to view
            fm.setWriter(parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "test_for_addition" + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file"), particles);
            fm.write();

         }
    }
    if (comm->getRank()==0) parameters->print();
    Teuchos::TimeMonitor::summarize();
    Kokkos::finalize();
#endif
return 0;
}

void writeToCSV(Compadre::EuclideanCoordsT& ec, Compadre::FieldT& f, std::ostream& os){
    os << "x, y, z, f," << std::endl;
    for (int j = 0; j < ec.nLocal(); ++j){
        const Compadre::XyzVector xyz = ec.getLocalCoords(j);
        const ST val = f.getLocalScalarVal(j);
        os << xyz.x << ", " << xyz.y << ", " << xyz.z << ", " << val << ", \n";
    }
}
