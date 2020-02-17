#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_RemapManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_SphericalCoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <iostream>

#define NUMBER_OF_PTS_EACH_DIRECTION 21

using namespace Compadre;
typedef int LO;
typedef long GO;
typedef double ST;

int main (int argc, char* args[]) {

	Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::oblackholestream bstream;
	Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

	Kokkos::initialize(argc, args);
	
	Teuchos::RCP<Teuchos::Time> ParameterTime = Teuchos::TimeMonitor::getNewCounter ("Parameter Initialization");
	Teuchos::RCP<Teuchos::Time> MiscTime = Teuchos::TimeMonitor::getNewCounter ("Miscellaneous");
	Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");
	Teuchos::RCP<Teuchos::Time> CoordinateInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Coordinate Insertion Time");
	Teuchos::RCP<Teuchos::Time> ParticleInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Particle Insertion Time");

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
	// There is a balance between "neighbors needed multiplier" and "cutoff multiplier"
	// Their product must increase somewhat with "porder", but "neighbors needed multiplier"
	// requires more of an increment than "cutoff multiplier" to achieve the required threshold (the other staying the same)
	//*********

	{
		MiscTime->start();
		{
			//
			// REMAP TEST
			//

			typedef Compadre::EuclideanCoordsT CT;
		 	Teuchos::RCP<Compadre::ParticlesT> particles =
		 		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
		 	CT* coords = (CT*)particles->getCoords();

			ST h_size = 1.0/((ST)NUMBER_OF_PTS_EACH_DIRECTION-1);

		 	// Coordinate insertion
		 	{
				std::vector<Compadre::XyzVector> verts_to_insert;
				if (comm->getRank()==0) {
					for (LO i=0; i<NUMBER_OF_PTS_EACH_DIRECTION; i++) {
						for (LO j=0; j<NUMBER_OF_PTS_EACH_DIRECTION; j++) {
							for (LO k=0; k<NUMBER_OF_PTS_EACH_DIRECTION; k++) {
								verts_to_insert.push_back(Compadre::XyzVector(h_size*i, h_size*j, h_size*k));
							}
						}
					}
				}
				if (comm->getRank()==1) {
					for (LO i=0; i<NUMBER_OF_PTS_EACH_DIRECTION; i++) {
						for (LO j=0; j<NUMBER_OF_PTS_EACH_DIRECTION; j++) {
							for (LO k=0; k<NUMBER_OF_PTS_EACH_DIRECTION-1; k++) {
								verts_to_insert.push_back(Compadre::XyzVector(h_size*i, h_size*j, 1+h_size*(k+1)));
							}
						}
					}
				}
				MiscTime->stop();
				CoordinateInsertionTime->start();
				coords->insertCoords(verts_to_insert);
				CoordinateInsertionTime->stop();
				MiscTime->start();
		 	}
		 	particles->resetWithSameCoords(); // must be called because particles doesn't know about coordinate insertions

		 	// build halo data
			ST halo_size = h_size *
					( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
							+ (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));
			particles->zoltan2Initialize();
			particles->buildHalo(halo_size);


			std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "remap_coords_output.nc";
			Compadre::FileManager fm;
			fm.setWriter(output_filename, particles);
			fm.write();

			// register fields
			particles->getFieldManager()->createField(3,"velocity","m/s");
			particles->getFieldManager()->createField(1,"pressure","Pa");
			particles->getFieldManager()->createField(3,"acceleration","m/{s^2}");

			Compadre::SecondOrderBasis second_order_sol;
			Compadre::SineProducts sinx_sol;
			particles->getFieldManager()->getFieldByName("velocity")->
					localInitFromVectorFunction(&second_order_sol);
			particles->getFieldManager()->getFieldByName("pressure")->
					localInitFromScalarFunction(&second_order_sol);
			particles->getFieldManager()->getFieldByName("acceleration")->
					localInitFromVectorFunction(&sinx_sol);

			particles->getFieldManager()->updateFieldsHaloData();

			particles->createNeighborhood();


		 	Teuchos::RCP<Compadre::ParticlesT> new_particles =
		 		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
		 	CT* new_coords = (CT*)new_particles->getCoords();

		 	// Particles insertion (Requires interpolation/remap)
		 	{
				std::vector<Compadre::XyzVector> verts_to_insert;

//				// only works in serial, otherwise messed up by repartition
//				Compadre::pool_type pool(1234);
//				Compadre::generator_type rand_gen = pool.get_state();
//				if (comm->getRank()==0) {
//					for (LO i=0; i<NUMBER_OF_PTS_EACH_DIRECTION-1; i++) {
//						for (LO j=0; j<NUMBER_OF_PTS_EACH_DIRECTION-1; j++) {
//							for (LO k=0; k<NUMBER_OF_PTS_EACH_DIRECTION-1; k++) {
//								ST rand_x = 0.5*h_size*0.25*rand_gen.drand(-1, 1);
//								ST rand_y = 0.5*h_size*0.25*rand_gen.drand(-1, 1);
//								ST rand_z = 0.5*h_size*0.25*rand_gen.drand(-1, 1);
//								verts_to_insert.push_back(Compadre::XyzVector(h_size*(0.5+i)+rand_x, h_size*(0.5+j)+rand_y, h_size*(0.5+k)+rand_z));
//							}
//						}
//					}
//				}

				std::vector<ST> pmin = coords->boundingBoxMinOnProcessor();
				std::vector<ST> pmax = coords->boundingBoxMaxOnProcessor();
				Compadre::pool_type pool(1234);
				Compadre::generator_type rand_gen = pool.get_state();

				ST eps = 1.0e-6;
				for (LO i=0; i<NUMBER_OF_PTS_EACH_DIRECTION*NUMBER_OF_PTS_EACH_DIRECTION; i++) {
					ST x = pmin[0] + rand_gen.drand(eps, (pmax[0]-pmin[0])-eps);
//					std::cout << "x between: " << pmin[0] << " and " << pmax[0] << " on proc# " << comm->getRank() << std::endl;
					ST y = pmin[1] + rand_gen.drand(eps, (pmax[1]-pmin[1])-eps);
//					std::cout << "y between: " << pmin[1] << " and " << pmax[1] << " on proc# " << comm->getRank() << std::endl;
					ST z = pmin[2] + rand_gen.drand(eps, (pmax[2]-pmin[2])-eps);
//					std::cout << "z between: " << pmin[2] << " and " << pmax[2] << " on proc# " << comm->getRank() << std::endl;
					verts_to_insert.push_back(Compadre::XyzVector(x, y, z));
//					std::cout << "inserted: (" << x << ", " << y << ", " << z << ")" << std::endl;
				}

				new_coords->insertCoords(verts_to_insert);
				new_particles->resetWithSameCoords();

				Teuchos::RCP<Compadre::RemapManager> rm = Teuchos::rcp(new Compadre::RemapManager(parameters, particles.getRawPtr(), new_particles.getRawPtr(), halo_size));
				Compadre::RemapObject ro("acceleration");
				rm->add(ro);
				rm->execute();

				MiscTime->stop();
				ParticleInsertionTime->start();
				particles->insertParticles(verts_to_insert);
				ParticleInsertionTime->stop();
				NormTime->start();
		 	}
			output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "only_acceleration_output.nc";
			fm.setWriter(output_filename, new_particles);
			fm.write();

		 	// point by point check to see how far off from correct value
			// check solution
		 	ST acceleration_weighted_l2_norm = 0;
		 	ST velocity_weighted_l2_norm = 0;
		 	ST pressure_weighted_l2_norm = 0;
			for( int j =0; j<coords->nLocal(); j++){
				Compadre::XyzVector xyz = coords->getLocalCoords(j);

				std::vector<ST> computed_acceleration = particles->getFieldManagerConst()->getFieldByName("acceleration")->getLocalVectorVal(j);
				Compadre::XyzVector exact_acceleration_xyz = sinx_sol.evalVector(xyz);
				std::vector<ST> exact_acceleration(3);
				exact_acceleration_xyz.convertToStdVector(exact_acceleration);
				for (LO i=0; i<3; i++) {
					acceleration_weighted_l2_norm += (computed_acceleration[i] - exact_acceleration[i])*(computed_acceleration[i] - exact_acceleration[i]);
				}

				std::vector<ST> computed_velocity = particles->getFieldManagerConst()->getFieldByName("velocity")->getLocalVectorVal(j);
				Compadre::XyzVector exact_velocity_xyz = second_order_sol.evalVector(xyz);
				std::vector<ST> exact_velocity(3);
				exact_velocity_xyz.convertToStdVector(exact_velocity);
				for (LO i=0; i<3; i++) {
					velocity_weighted_l2_norm += (computed_velocity[i] - exact_velocity[i])*(computed_velocity[i] - exact_velocity[i]);
				}

				ST computer_pressure = particles->getFieldManagerConst()->getFieldByName("pressure")->getLocalScalarVal(j);
				ST exact_pressure = second_order_sol.evalScalar(xyz);
				pressure_weighted_l2_norm += (computer_pressure - exact_pressure)*(computer_pressure - exact_pressure);
			}
			acceleration_weighted_l2_norm = sqrt(acceleration_weighted_l2_norm) /((ST)coords->nGlobalMax());
			velocity_weighted_l2_norm = sqrt(velocity_weighted_l2_norm) /((ST)coords->nGlobalMax());
			pressure_weighted_l2_norm = sqrt(pressure_weighted_l2_norm) /((ST)coords->nGlobalMax());

			ST global_norm_acceleration, global_norm_velocity, global_norm_pressure;

			Teuchos::Ptr<ST> global_norm_ptr_acceleration(&global_norm_acceleration);
			Teuchos::Ptr<ST> global_norm_ptr_velocity(&global_norm_velocity);
			Teuchos::Ptr<ST> global_norm_ptr_pressure(&global_norm_pressure);

			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, acceleration_weighted_l2_norm, global_norm_ptr_acceleration);
			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, velocity_weighted_l2_norm, global_norm_ptr_velocity);
			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, pressure_weighted_l2_norm, global_norm_ptr_pressure);

			if (comm->getRank()==0) std::cout << "\n\n\n\nGlobal Norm Acceleration: " << global_norm_acceleration << "\n";
			if (comm->getRank()==0) std::cout << "Global Norm Velocity: " << global_norm_velocity << "\n";
			if (comm->getRank()==0) std::cout << "Global Norm Pressure: " << global_norm_pressure << "\n\n\n\n\n";

			output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "remap_particle_output.nc";
			fm.setWriter(output_filename, particles);
			fm.write();

			TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_acceleration > 1e-6, "Acceleration error too large.");
			TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_velocity > 1e-15, "Velocity error too large (should be exact).");
			TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_pressure > 1e-15, "Pressure error too large (should be exact).");
			NormTime->stop();
		 }
	}
	if (comm->getRank()==0) parameters->print();
	Teuchos::TimeMonitor::summarize();
	Kokkos::finalize();
return 0;
}

