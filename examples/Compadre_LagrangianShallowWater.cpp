#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ProblemExplicitTransientT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_SphericalCoordsT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <Compadre_LagrangianShallowWater_Operator.hpp>
#include <Compadre_LagrangianShallowWater_Sources.hpp>
#include <Compadre_LagrangianShallowWater_BoundaryConditions.hpp>

#include "Compadre_GlobalConstants.hpp"
static const Compadre::GlobalConstants consts;

#include <Compadre_GMLS.hpp> // for getNP()

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::XyzVector xyz_type;


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

	Teuchos::RCP<Teuchos::Time> SphericalParticleTime = Teuchos::TimeMonitor::getNewCounter ("Spherical Particle Time");
	Teuchos::RCP<Teuchos::Time> NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Neighbor Search Time");
	Teuchos::RCP<Teuchos::Time> FirstReadTime = Teuchos::TimeMonitor::getNewCounter ("1st Read Time");
	Teuchos::RCP<Teuchos::Time> AssemblyTime = Teuchos::TimeMonitor::getNewCounter ("Assembly Time");
	Teuchos::RCP<Teuchos::Time> SolvingTime = Teuchos::TimeMonitor::getNewCounter ("Solving Time");
	Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");
	Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");

	{

		typedef Compadre::EuclideanCoordsT CT;
		Teuchos::RCP<Compadre::ParticlesT> particles =
				Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));

		//Read in data file.
		FirstReadTime->start();
		Compadre::FileManager fm;
		std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));
		fm.setReader(testfilename, particles, parameters->get<Teuchos::ParameterList>("io").get<std::string>("nc type"));
        fm.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
        fm.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
		fm.read();
		FirstReadTime->stop();

		ST halo_size;
		particles->getFieldManagerConst()->listFields(std::cout);


		// resize to a sphere of Earth's radius

		Compadre::ShallowWaterTestCases sw_function = Compadre::ShallowWaterTestCases(parameters->get<int>("shallow water test case"), 0 /*alpha*/);
		sw_function.setAtmosphere();

 		Compadre::ConstantEachDimension zero_function = Compadre::ConstantEachDimension(0,0,0);


		particles->getCoords()->getPts(false /*not halo*/, true)->scale(sw_function.getRadius());

		{
			const ST h_size = .0833333*sw_function.getRadius();

			particles->zoltan2Initialize(); // also applies the repartition to coords, flags, and fields
			if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
				halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
			} else {
				halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size")*sw_function.getRadius();
			}
			particles->buildHalo(halo_size);


//	 		particles->getFieldManager()->listFields(std::cout);

//			//TODO: register fields and set up initial conditions
//	 		// this tests adding fields after generating the partition and halo information
	 		particles->getFieldManager()->createField(3, "velocity", "m/s");
	 		particles->getFieldManager()->createField(1, "height", "m");


			particles->getFieldManager()->getFieldByName("velocity")->
					localInitFromVectorFunction(&sw_function);
			particles->getFieldManager()->getFieldByName("height")->
					localInitFromScalarFunction(&sw_function);
			if (parameters->get<int>("shallow water test case") == 2) {
				particles->getFieldManager()->getFieldByName("height")->
						scale(1./sw_function.getGravity());
			} else {

//				particles->getFieldManager()->getFieldByName("height")->
//						scale(-1.);
//		 		Compadre::host_view_type h_data = particles->getFieldManager()->getFieldByName("height")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();
//		 		for (LO i=0; i<particles->getCoordsConst()->nLocal(); ++i) {
//	 				h_data(i,0) += 5960;
//		 		}

			}

			Compadre::ConstantEachDimension proc_func = Compadre::ConstantEachDimension(comm->getRank(),1,1);
			particles->getFieldManager()->createField(1, "processor", "none");
			particles->getFieldManager()->getFieldByName("processor")->
					localInitFromScalarFunction(&proc_func);

//			particles->getFieldManager()->createField(3, "reference velocity", "m/s");
//			particles->getFieldManager()->getFieldByName("reference velocity")->
//					localInitFromVectorFunction(&sw_function);
//
//	 		particles->getFieldManager()->createField(3, "original velocity", "m/s");
//			particles->getFieldManager()->getFieldByName("original velocity")->
//					localInitFromVectorFunction(&sw_function);
//
//	 		particles->getFieldManager()->createField(1, "reference height", "m");
//			particles->getFieldManager()->getFieldByName("reference height")->
//					localInitFromScalarFunction(&sw_function);
//			particles->getFieldManager()->getFieldByName("reference height")->
//					scale(1./sw_function.getGravity());
//
//	 		particles->getFieldManager()->createField(1, "original height", "m");
//			particles->getFieldManager()->getFieldByName("original height")->
//					localInitFromScalarFunction(&sw_function);
//			particles->getFieldManager()->getFieldByName("original height")->
//					scale(1./sw_function.getGravity());



//			particles->getFieldManager()->updateFieldsHaloData();

//	 		particles->getFieldManager()->listFields(std::cout);
//
//	 		Compadre::ConstantEachDimension function1 = Compadre::ConstantEachDimension(1,1,1);
//	 		Compadre::ConstantEachDimension function2 = Compadre::ConstantEachDimension(0,0,0);
//			particles->getFieldManager()->getFieldByName("velocity")->
//					localInitFromVectorFunction(&function1);
//			particles->getFieldManager()->getFieldByName("height")->
//					localInitFromVectorFunction(&function2);
//			particles->getFieldManager()->updateFieldsHaloData();

			//TODO: register fields and set up initial conditions
	 		// this tests adding fields after generating the partition and halo information
//	 		particles->getFieldManager()->createField(3, "velocity", "m/s");
//	 		particles->getFieldManager()->createField(3, "original locations", "m/s");
//	 		const Teuchos::RCP<Compadre::mvec_type> material_point_vector = particles->getCoordsConst()->getPts(false, false);// not halo, material coords
//	 		Teuchos::RCP<Compadre::mvec_type> field_vector = particles->getFieldManager()->getFieldByName("original locations")->getMultiVectorPtr();
//
//	 		Compadre::host_view_type material_point_data = material_point_vector->getLocalView<Compadre::host_view_type>();
//	 		Compadre::host_view_type field_data = field_vector->getLocalView<Compadre::host_view_type>();
//
//	 		for (LO i=0; i<particles->getCoordsConst()->nLocal(); ++i) {
//	 			for (LO j=0; j<3; ++j) {
//	 				field_data(i,j) = material_point_data(i,j);
//	 			}
//	 		}
//	 		particles->getFieldManager()->createField(3, "new locations", "m/s");



	 		particles->getFieldManager()->createField(3, "displacement", "m/s");
			particles->getFieldManager()->getFieldByName("displacement")->
					localInitFromVectorFunction(&zero_function);

//			particles->getFieldManager()->updateFieldsHaloData();
//			{
//				// set displacements to have initial values of coordinates in reference domain
//				Compadre::host_view_type initial_coords = particles->getCoords()->getPts()->getLocalView<Compadre::host_view_type>();
//				Teuchos::RCP<Compadre::FieldT> disp_field = particles->getFieldManager()->getFieldByName("displacement");
//
//				for (LO i=0; i<initial_coords.extent(0); ++i) {
//					for (LO j=0; j<3; ++j) {
//						disp_field->setLocalScalarVal(i,j,initial_coords(i,j));
//					}
//				}
//			}

//			particles->getFieldManager()->createField(3, "test1", "m/s");
//			particles->getFieldManager()->createField(1, "test2", "m/s");


			particles->getFieldManager()->updateFieldsHaloData();


//						WriteTime->start();
//						{
//							std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//							Compadre::FileManager fm;
//							fm.setWriter(output_filename, particles);
//							fm.write();
//						}
//						WriteTime->stop();

			particles->createDOFManager();
			particles->getDOFManager()->generateDOFMap();

			NeighborSearchTime->start();
			const ST h_support = parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size")*sw_function.getRadius();

			particles->createNeighborhood();
			particles->getNeighborhood()->setAllHSupportSizes(h_support);

			LO neighbors_needed = Compadre::GMLS::getNP(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 2);
 		    particles->getNeighborhood()->constructAllNeighborLists(particles->getCoordsConst()->getHaloSize(),
                parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
                true /*dry run for sizes*/,
                neighbors_needed,
                parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
                parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
                parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
                parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));

			double h_min = particles->getNeighborhood()->computeMinHSupportSize(true /*global min*/);

			NeighborSearchTime->stop();

			Teuchos::RCP<Compadre::ProblemExplicitTransientT> problem =
				Teuchos::rcp( new Compadre::ProblemExplicitTransientT(particles));

			// construct physics, sources, and boundary conditions
			Teuchos::RCP<Compadre::LagrangianShallowWaterPhysics> physics =
				Teuchos::rcp( new Compadre::LagrangianShallowWaterPhysics(particles));
			Teuchos::RCP<Compadre::LagrangianShallowWaterSources> source =
				Teuchos::rcp( new Compadre::LagrangianShallowWaterSources(particles));
			Teuchos::RCP<Compadre::LagrangianShallowWaterBoundaryConditions> bcs =
				Teuchos::rcp( new Compadre::LagrangianShallowWaterBoundaryConditions(particles));


			physics->setGravity(sw_function.getGravity());
			if (comm->getRank()==0)
				printf("Gravity: %f\n", sw_function.getGravity());

			// set physics, sources, and boundary conditions in the problem
			problem->setPhysics(physics);
			problem->setSources(source);
			problem->setBCS(bcs);

			// assembly
			AssemblyTime->start();
			problem->initialize();
			AssemblyTime->stop();

			// get stable timestep
			double stable_timestep_size;
			if (parameters->get<int>("shallow water test case") == 2) {
				stable_timestep_size = (parameters->get<Teuchos::ParameterList>("time").get<double>("dt") > 0) ?
					parameters->get<Teuchos::ParameterList>("time").get<double>("dt") : physics->getStableTimeStep(h_min, (2.94e+4)/sw_function.getGravity())*parameters->get<Teuchos::ParameterList>("time").get<double>("dt multiplier");
			} else {
				stable_timestep_size = (parameters->get<Teuchos::ParameterList>("time").get<double>("dt") > 0) ?
					parameters->get<Teuchos::ParameterList>("time").get<double>("dt") : physics->getStableTimeStep(h_min, 5960)*parameters->get<Teuchos::ParameterList>("time").get<double>("dt multiplier");
			}

//			double stable_timestep_size = physics->getStableTimeStep(h_min, (2.94e+4)/sw_function.getGravity());
			TEUCHOS_TEST_FOR_EXCEPT_MSG(stable_timestep_size==0, "Stable timestep not found or dt specified to 0.");

			//solving
			SolvingTime->start();
			problem->solve(parameters->get<Teuchos::ParameterList>("time").get<int>("rk order") /* rk 4*/, parameters->get<Teuchos::ParameterList>("time").get<double>("t0"), parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"), stable_timestep_size, parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file"));
			SolvingTime->stop();


//			{
//				// set displacements to have initial values of coordinates in reference domain
//				Compadre::host_view_type initial_coords = particles->getCoords()->getPts(false /*halo*/, false /*use_physical_coords*/)->getLocalView<Compadre::host_view_type>();
//				Teuchos::RCP<Compadre::FieldT> disp_field = particles->getFieldManager()->getFieldByName("displacement");
//
//				for (LO i=0; i<initial_coords.extent(0); ++i) {
//					for (LO j=0; j<2; ++j) {
//						printf("diff in (%d,%d) is: %f, %f, %f\n", i, j, initial_coords(i,j)-disp_field->getLocalScalarVal(i,j), initial_coords(i,j), disp_field->getLocalScalarVal(i,j));
//					}
//				}
//			}

//	 		const Teuchos::RCP<Compadre::mvec_type> physical_point_vector = particles->getCoordsConst()->getPts(false, false);// not halo, material coords
//	 		field_vector = particles->getFieldManager()->getFieldByName("new locations")->getMultiVectorPtr();
//
//	 		Compadre::host_view_type physical_point_data = physical_point_vector->getLocalView<Compadre::host_view_type>();
//	 		field_data = field_vector->getLocalView<Compadre::host_view_type>();
//	 		for (LO i=0; i<particles->getCoordsConst()->nLocal(); ++i) {
//	 			for (LO j=0; j<3; ++j) {
//	 				field_data(i,j) = physical_point_data(i,j);
//	 			}
//	 		}

//			WriteTime->start();
//			{
//				std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//				Compadre::FileManager fm;
//				fm.setWriter(output_filename, particles);
//				fm.write();
//			}
//			WriteTime->stop();

		 	// point by point check to see how far off from correct value
			// check solution

			NormTime->start();
		 	ST velocity_error_weighted_l2_norm = 0;
		 	ST exact_velocity_weighted_l2_norm = 0;

		 	ST height_error_weighted_l2_norm = 0;
		 	ST exact_height_weighted_l2_norm = 0;

		 	CT* coords = (CT*)particles->getCoords();
//		 	double end_time = parameters->get<Teuchos::ParameterList>("time").get<double>("t_end");



			if (parameters->get<int>("shallow water test case") == 2) {

			Compadre::host_view_type initial_coords = particles->getCoords()->getPts(false /*halo*/, false /*use_physical_coords*/)->getLocalView<Compadre::host_view_type>();
			Compadre::host_view_type disp_field = particles->getFieldManager()->getFieldByName("velocity")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
			Compadre::host_view_type height_field = particles->getFieldManager()->getFieldByName("height")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
			Compadre::host_view_type grid_area_field = particles->getFieldManager()->getFieldByName("grid_area")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();

			for(int j =0; j<coords->nLocal(); j++){

				xyz_type xyz = coords->getLocalCoords(j, true /* use_halo */, true);
				xyz_type exact = sw_function.evalVector(xyz);
				ST exact_height = sw_function.evalScalar(xyz)/sw_function.getGravity();

				velocity_error_weighted_l2_norm += (disp_field(j,0) - exact.x)*(disp_field(j,0) - exact.x)*grid_area_field(j,0);
				velocity_error_weighted_l2_norm += (disp_field(j,1) - exact.y)*(disp_field(j,1) - exact.y)*grid_area_field(j,0);
				velocity_error_weighted_l2_norm += (disp_field(j,2) - exact.z)*(disp_field(j,2) - exact.z)*grid_area_field(j,0);
				height_error_weighted_l2_norm += (height_field(j,0) - exact_height)*(height_field(j,0) - exact_height)*grid_area_field(j,0);

//				physical_coordinate_weighted_l2_norm += (disp_field(j,0) - (initial_coords(j,0)*std::cos(2*PI*end_time) + initial_coords(j,1)*std::sin(2*PI*end_time)))*(disp_field(j,0) - (initial_coords(j,0)*std::cos(2*PI*end_time) + initial_coords(j,1)*std::sin(2*PI*end_time)))*grid_area_field(j,0);
//				physical_coordinate_weighted_l2_norm += (disp_field(j,1) - (initial_coords(j,1)*std::cos(2*PI*end_time) - initial_coords(j,0)*std::sin(2*PI*end_time)))*(disp_field(j,1) - (initial_coords(j,1)*std::cos(2*PI*end_time) - initial_coords(j,0)*std::sin(2*PI*end_time)))*grid_area_field(j,0);
//				physical_coordinate_weighted_l2_norm += (disp_field(j,2) - initial_coords(j,2))*(disp_field(j,2) - initial_coords(j,2))*grid_area_field(j,0);

				exact_velocity_weighted_l2_norm += exact.x*exact.x*grid_area_field(j,0);
				exact_velocity_weighted_l2_norm += exact.y*exact.y*grid_area_field(j,0);
				exact_velocity_weighted_l2_norm += exact.z*exact.z*grid_area_field(j,0);
				exact_height_weighted_l2_norm += exact_height*exact_height*grid_area_field(j,0);

			}

			{
				// velocity error communication
				ST global_norm_velocity_error;
				ST global_norm_exact_velocity;
				Teuchos::Ptr<ST> global_norm_ptr_velocity_error(&global_norm_velocity_error);
				Teuchos::Ptr<ST> global_norm_ptr_exact_velocity(&global_norm_exact_velocity);

				Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, velocity_error_weighted_l2_norm, global_norm_ptr_velocity_error);
				Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, exact_velocity_weighted_l2_norm, global_norm_ptr_exact_velocity);

				if (comm->getRank()==0) std::cout << "Global relative VELOCITY error: " << std::sqrt(global_norm_velocity_error) /  std::sqrt(global_norm_exact_velocity)  << "\n";
			}
			{
				// height error communication
				ST global_norm_height_error;
				ST global_norm_exact_height;
				Teuchos::Ptr<ST> global_norm_ptr_height_error(&global_norm_height_error);
				Teuchos::Ptr<ST> global_norm_ptr_exact_height(&global_norm_exact_height);

				Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, height_error_weighted_l2_norm, global_norm_ptr_height_error);
				Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, exact_height_weighted_l2_norm, global_norm_ptr_exact_height);

				if (comm->getRank()==0) std::cout << "Global relative HEIGHT error: " << std::sqrt(global_norm_height_error) /  std::sqrt(global_norm_exact_height)  << "\n\n\n\n";
			}

			} else {
				std::cout << "Test case 5 complete." << std::endl;
			}
			NormTime->stop();
		}
	}

//	if (comm->getRank()==0) parameters->print();
	Teuchos::TimeMonitor::summarize();
	Kokkos::finalize();
return 0;
}

