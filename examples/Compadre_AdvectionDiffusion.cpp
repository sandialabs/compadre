#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_nanoflannInformation.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>
#include <Compadre_RemapManager.hpp>

#include <Compadre_GMLS.hpp> // for getNP()

#include <Compadre_AdvectionDiffusion_Sources.hpp>
#include <Compadre_AdvectionDiffusion_BoundaryConditions.hpp>
#include <Compadre_AdvectionDiffusion_Operator.hpp>

#define STACK_TRACE(call) try { call; } catch (const std::exception& e ) { TEUCHOS_TRACE(e); }

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::XyzVector xyz_type;
typedef Compadre::EuclideanCoordsT CT;

using namespace Compadre;

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
	
	Teuchos::RCP<Teuchos::Time> FirstReadTime = Teuchos::TimeMonitor::getNewCounter ("1st Read Time");
	Teuchos::RCP<Teuchos::Time> AssemblyTime = Teuchos::TimeMonitor::getNewCounter ("Assembly Time");
	Teuchos::RCP<Teuchos::Time> SolvingTime = Teuchos::TimeMonitor::getNewCounter ("Solving Time");
	Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");
	Teuchos::RCP<Teuchos::Time> SecondReadTime = Teuchos::TimeMonitor::getNewCounter ("2nd Read Time");

	// this proceeds setting up the problem so that the parameters will be propagated down into the physics and bcs
	try {
		parameters->get<std::string>("solution type");
	} catch (Teuchos::Exceptions::InvalidParameter & exception) {
		parameters->set<std::string>("solution type","sine");
	}

	{
		std::vector<std::string> fnames(5);
		std::vector<double> hsize(5);
		std::vector<double> errors(5);
		const std::string testfilename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + "/" + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file");

		//
		// GMLS Test
		//
		int Porder = parameters->get<Teuchos::ParameterList>("remap").get<int>("porder");
		double h_size = 1./12.;

		Teuchos::RCP<Compadre::ParticlesT> cells =
				Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
		const CT * coords = (CT*)(cells->getCoordsConst());

		//Read in data file. Needs to be a file with one scalar field.
		FirstReadTime->start();
		Compadre::FileManager fm;
		fm.setReader(testfilename, cells);
		fm.read();
		FirstReadTime->stop();

		cells->zoltan2Initialize();


		ST halo_size;
		{
			if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
				halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
			} else {
				halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
			}
 			cells->buildHalo(halo_size);
 			cells->createDOFManager();



			//Set the radius for the neighbor list:
	 		ST h_support;
	 		if (parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("dynamic radius")) {
	 			h_support = h_size;
	 		} else {
	 			h_support = parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size");
	 		}
	 		cells->createNeighborhood();
	 		cells->getNeighborhood()->setAllHSupportSizes(h_support);

	 		LO neighbors_needed = Compadre::GMLS::getNP(Porder, 2 /*dimension*/);

			LO extra_neighbors = parameters->get<Teuchos::ParameterList>("remap").get<double>("neighbors needed multiplier") * neighbors_needed;
			cells->getNeighborhood()->constructAllNeighborList(cells->getCoordsConst()->getHaloSize(), extra_neighbors);

		}
//		{
//	        WriteTime->start();
//	        std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//	        std::string writetest_output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "writetest" + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//	        fm.setWriter(output_filename, cells);
//	        if (parameters->get<Teuchos::ParameterList>("io").get<bool>("vtk produce mesh")) fm.generateWriteMesh();
//	        fm.write();
//	        WriteTime->stop();
//		}
		Teuchos::RCP<Compadre::ParticlesT> particles =
				Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
        auto particle_coords = Teuchos::rcpFromRef<CT>(*(CT*)cells->getCoords());
        particles->setCoords(particle_coords);

		particles->getFieldManager()->createField(1, "solution", "m/s");
        particles->createDOFManager();
        particles->getDOFManager()->generateDOFMap();
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

	 		LO neighbors_needed = Compadre::GMLS::getNP(Porder, 2 /*dimension*/);

			LO extra_neighbors = parameters->get<Teuchos::ParameterList>("remap").get<double>("neighbors needed multiplier") * neighbors_needed;
			particles->getNeighborhood()->constructAllNeighborList(particles->getCoordsConst()->getHaloSize(), extra_neighbors);

		}

        // Iterative solver for the problem
        Teuchos::RCP<Compadre::ProblemT> problem = Teuchos::rcp( new Compadre::ProblemT(particles));

        // construct physics, sources, and boundary conditions
        Teuchos::RCP<Compadre::AdvectionDiffusionPhysics> physics =
          Teuchos::rcp( new Compadre::AdvectionDiffusionPhysics(particles, Porder));
        Teuchos::RCP<Compadre::AdvectionDiffusionSources> source =
            Teuchos::rcp( new Compadre::AdvectionDiffusionSources(particles));
        Teuchos::RCP<Compadre::AdvectionDiffusionBoundaryConditions> bcs =
            Teuchos::rcp( new Compadre::AdvectionDiffusionBoundaryConditions(particles));

        // set physics, sources, and boundary conditions in the problem
        xyz_type advection_field(1,1,1);

        // set advection and diffusion for physics
        physics->setAdvectionField(advection_field);
        physics->setDiffusion(parameters->get<Teuchos::ParameterList>("physics").get<double>("diffusion"));

        physics->setCells(cells);
        problem->setPhysics(physics);
        source->setPhysics(physics);

        problem->setSources(source);
        problem->setBCS(bcs);

        bcs->flagBoundaries();

        // assembly
        AssemblyTime->start();
        problem->initialize();
        AssemblyTime->stop();

        //solving
        SolvingTime->start();
        problem->solve();
        SolvingTime->stop();
        particles->getFieldManager()->updateFieldsHaloData();


		//Teuchos::RCP<Compadre::RemapManager> rm = Teuchos::rcp(new Compadre::RemapManager(parameters, particles.getRawPtr(), cells.getRawPtr(), halo_size));
		//Compadre::RemapObject ro("solution", "another_solution", TargetOperation::ScalarPointEvaluation);
		//rm->add(ro);
		//STACK_TRACE(rm->execute());
//
//			// write matrix and rhs to files
////				Tpetra::MatrixMarket::Writer<Compadre::mvec_type>::writeDenseFile("b"+std::to_string(i), *problem->getb(), "rhs", "description");
////				Tpetra::MatrixMarket::Writer<Compadre::crs_matrix_type>::writeSparseFile("A"+std::to_string(i), problem->getA(), "A", "description");
//		}
//
//
//		Teuchos::RCP< Compadre::FieldT > PField = particles->getFieldManagerConst()->getFieldByID(0);
//
		// check solution
		double norm = 0.0;
		double exact = 0.0;

		Teuchos::RCP<Compadre::AnalyticFunction> function;
		function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts(2 /*dimension*/)));

		particles->getFieldManager()->createField(1, "exact solution", "m/s");
		auto exact_view = particles->getFieldManager()->getFieldByName("exact solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();
		for( int j =0; j<coords->nLocal(); j++){
			xyz_type xyz = coords->getLocalCoords(j);
			const ST val = particles->getFieldManagerConst()->getFieldByName("solution")->getLocalScalarVal(j);
			exact = function->evalScalar(xyz);
			norm += (exact - val)*(exact-val);
			exact_view(j,0) = exact;
		}
		norm /= (double)(coords->nGlobalMax());

		ST global_norm;
		Teuchos::Ptr<ST> global_norm_ptr(&global_norm);
		Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, norm, global_norm_ptr);
		global_norm = sqrt(global_norm);
		if (comm->getRank()==0) std::cout << "Global Norm: " << global_norm << "\n";


        {
            WriteTime->start();
            std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
            fm.setWriter(output_filename, particles);
            if (parameters->get<Teuchos::ParameterList>("io").get<bool>("vtk produce mesh")) fm.generateWriteMesh();
            fm.write();
            WriteTime->stop();
        }
        //{
        //    WriteTime->start();
        //    std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
        //    fm.setWriter(output_filename, cells);
        //    if (parameters->get<Teuchos::ParameterList>("io").get<bool>("vtk produce mesh")) fm.generateWriteMesh();
        //    fm.write();
        //    WriteTime->stop();
        //}
//		errors[i] = global_norm;
//
//		WriteTime->start();
//		std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//		std::string writetest_output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "writetest" + std::to_string(i) /* loop # */ + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//		fm.setWriter(output_filename, particles);
//		if (parameters->get<Teuchos::ParameterList>("io").get<bool>("vtk produce mesh")) fm.generateWriteMesh();
//		fm.write();
//		WriteTime->stop();
//		{
//			//
//			// VTK File Reader Test and Parallel VTK File Re-Reader
//			//
//
//			// read solution back in which should also reset flags
//
//			SecondReadTime->start();
//			Teuchos::RCP<Compadre::ParticlesT> test_particles =
//					Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
//
//			fm.setReader(output_filename, test_particles);
//			fm.read();
//
//			fm.setWriter(writetest_output_filename, test_particles);
//			fm.write();
//			SecondReadTime->stop();
//
//			test_particles->getFieldManagerConst()->listFields(std::cout);
//
//		}
//
//		if (parameters->get<Teuchos::ParameterList>("solver").get<std::string>("type")=="direct")
//			Teuchos::TimeMonitor::summarize();
//
//		TEUCHOS_TEST_FOR_EXCEPT_MSG(errors[i]!=errors[i], "NaN found in error norm.");
//		if (parameters->get<std::string>("solution type")=="sine") {
//			 if (i>0) TEUCHOS_TEST_FOR_EXCEPT_MSG(errors[i-1]/errors[i] < 3.5, "Second order not achieved for sine solution (should be 4).");
//		} else {
//			TEUCHOS_TEST_FOR_EXCEPT_MSG(errors[i] > 1e-13, "Second order solution not recovered exactly.");
//		}
//		if (comm->getRank()==0) parameters->print();
	}
	 Teuchos::TimeMonitor::summarize();
	Kokkos::finalize();
	#endif
	return 0;
}


