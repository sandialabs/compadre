#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Compadre_GlobalConstants.hpp>
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

typedef int LO;
typedef long GO;
typedef double ST;

int main (int argc, char* args[]) {

	Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::oblackholestream bstream;
	Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

	Kokkos::initialize(argc, args);
	
//	Teuchos::RCP<Teuchos::Time> Part = Teuchos::TimeMonitor::getNewCounter ("PartTime");

	Teuchos::RCP<Teuchos::Time> ParameterTime = Teuchos::TimeMonitor::getNewCounter ("Parameter Initialization");
	Teuchos::RCP<Teuchos::Time> MiscTime = Teuchos::TimeMonitor::getNewCounter ("Miscellaneous");
	Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");
	Teuchos::RCP<Teuchos::Time> CoordinateInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Coordinate Insertion Time");
	Teuchos::RCP<Teuchos::Time> ParticleInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Particle Insertion Time");
	Teuchos::RCP<Teuchos::Time> NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Lagrangian - Neighbor Search Time");

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

	TEUCHOS_TEST_FOR_EXCEPT_MSG(comm->getSize()!=2, "Example only built to be run on two processors.");

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
			particles->zoltan2Initialize(false /*partition using material coords*/);
			particles->buildHalo(halo_size, false /*build halo using material coords*/);




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

			// should add fn to Lagrangian coords and set as physical coords
			Compadre::ScaleOfEachDimension xy_inc_fn(1,1,0);
			particles->getCoords()->deltaUpdatePhysicalCoordsFromVectorFunction(&xy_inc_fn);
//			particles->getCoords()->resetPhysicalCoords(); // can be performed to see if reconstruction of physical is reconstruction of lagrangian

			// writes out a 1x1x2 shaped rectangle
			std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "lagrange_coords_post_transformation.nc";
			Compadre::FileManager fm;
			parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","lagrangian");
			fm.setWriter(output_filename, particles);
			fm.write();

			// writes out a 2x2x2 shaped rectangle by expanding in the x and y directions by x and y
			output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "physical_coords_post_transformation.nc";
			parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","physical");
			fm.setWriter(output_filename, particles);
			fm.write();




			if (comm->getRank()==0) std::cout << "Insert test 1 (Inserted as lagrangian, physical derived through interpolation): " << std::endl;

			{
				// 1. This test creates a new particle set and remaps values onto it from the previous set
				// It uses the Lagrangian coordinates to get coefficients and neighborhoods from which to interpolate
				// the physical coordinates. At the end, it has physical coordinates that should be interpolated from the prior set.

				// Particles insertion (Requires interpolation/remap)
				Teuchos::RCP<Compadre::ParticlesT> new_particles =
					Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
				CT* new_coords = (CT*)new_particles->getCoords();
				{
					std::vector<Compadre::XyzVector> verts_to_insert;

					/*
					 * Zoltan2 partition was performed before the physical coordinates were transformed (i.e. Lagrangian neighborhoods)
					 * Now that the physical coordinates are transformed, we insert the new coordinates into the physical frame
					 * (i.e. they need to construct Lagrangian coordinates as well through remap)
					 *
					 * But for now, pmin and pmax are determined using bounding boxes before the transformation
					 */
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

					new_particles->resetWithSameCoords();
					for( int j =0; j<new_coords->nLocal(); j++){
						Compadre::XyzVector xyz = new_coords->getLocalCoords(j, false /*halo*/, false);
						ST diff = 0;
						diff += abs(verts_to_insert[j].x - xyz.x);diff += abs(verts_to_insert[j].y - xyz.y);diff += abs(verts_to_insert[j].z - xyz.z);
						if (diff>0) {
						std::cout << xyz.x << " " << xyz.y << " " << xyz.z << std::endl;
						std::cout << verts_to_insert[j].x << " " << verts_to_insert[j].y << " " << verts_to_insert[j].z << std::endl;
						}
					}

					Teuchos::RCP<Compadre::RemapManager> rm = Teuchos::rcp(new Compadre::RemapManager(parameters, particles.getRawPtr(), new_particles.getRawPtr(), halo_size));
					Compadre::RemapObject ro(-21); // reserved for interpolating physical coords from coefficients based on Lagrangian coords
					Compadre::RemapObject ra("acceleration");
					rm->add(ro);
					rm->add(ra);

					rm->execute(false /* don't keep neighborhoods */, false /* don't keep GMLS */, false /* don't reuse neighborhoods */, false /* don't reuse GMLS */, false /* use material coords to form neighborhoods */);

					MiscTime->stop();
					NormTime->start();
					std::vector<Compadre::XyzVector> verts_screwed_up;
					{
					 	// point by point check to see how far off from correct value
						// check solution
					 	ST acceleration_weighted_l2_norm = 0;
					 	ST physical_coordinate_weighted_l2_norm = 0;
						for( int j =0; j<new_coords->nLocal(); j++){
							Compadre::XyzVector xyz = new_coords->getLocalCoords(j, false /*halo*/, false /*use_physical_coords*/); // make sure this matches how they were assigned originally (as a function of Lagrangian coordinates or physical coordinates)

							std::vector<ST> computed_acceleration = new_particles->getFieldManagerConst()->getFieldByName("acceleration")->getLocalVectorVal(j);
							Compadre::XyzVector exact_acceleration_xyz = sinx_sol.evalVector(xyz);
							std::vector<ST> exact_acceleration(3);
							exact_acceleration_xyz.convertToStdVector(exact_acceleration);
							for (LO i=0; i<3; i++) {
								acceleration_weighted_l2_norm += (computed_acceleration[i] - exact_acceleration[i])*(computed_acceleration[i] - exact_acceleration[i]);
							}

							Compadre::XyzVector computed_xyz = new_coords->getLocalCoords(j);
							Compadre::XyzVector exact_physical_coord_xyz = xy_inc_fn.evalVector(xyz);
							exact_physical_coord_xyz.x += xyz.x;
							exact_physical_coord_xyz.y += xyz.y;
							exact_physical_coord_xyz.z += xyz.z;
							std::vector<ST> exact_physical_coord(3);
							std::vector<ST> computed_physical_coord(3);
							exact_physical_coord_xyz.convertToStdVector(exact_physical_coord);
							computed_xyz.convertToStdVector(computed_physical_coord);
							for (LO i=0; i<3; i++) {

								physical_coordinate_weighted_l2_norm += (computed_physical_coord[i] - exact_physical_coord[i])*(computed_physical_coord[i] - exact_physical_coord[i]);
								if ( (computed_physical_coord[i] - exact_physical_coord[i])*(computed_physical_coord[i] - exact_physical_coord[i])  > 1e-7) {
									if (computed_physical_coord[i] > exact_physical_coord[i]) {
										// note computed always less than physical
										std::cout << j << "," << i << " " << computed_physical_coord[i] << " " << exact_physical_coord[i] << std::endl;
									}
									verts_screwed_up.push_back(Compadre::XyzVector(xyz.x, xyz.y, xyz.z));
								}
							}
						}

//						{
//						// won't run if nothing is messed up
//							// temporary for diagnostic
//							Teuchos::RCP<Compadre::ParticlesT> new_particles2 =
//								Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
//							CT* new_coords2 = (CT*)new_particles2->getCoords();
//							if (parameters->get<std::string>("simulation type")=="lagrangian") {
//								new_coords2->insertCoords(verts_screwed_up);
//							}
//							new_particles2->resetWithSameCoords();
//
//							parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","lagrangian");
//							output_filename = "screwed_up_lagrangian.nc";
//							fm.setWriter(output_filename, new_particles2);
//							fm.write();
//						}

						acceleration_weighted_l2_norm = sqrt(acceleration_weighted_l2_norm) /((ST)new_coords->nGlobalMax());
						physical_coordinate_weighted_l2_norm = sqrt(physical_coordinate_weighted_l2_norm) /((ST)new_coords->nGlobalMax());

						ST global_norm_acceleration, global_norm_physical_coord;

						Teuchos::Ptr<ST> global_norm_ptr_acceleration(&global_norm_acceleration);
						Teuchos::Ptr<ST> global_norm_ptr_physical_coord(&global_norm_physical_coord);

						Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, acceleration_weighted_l2_norm, global_norm_ptr_acceleration);
						Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, physical_coordinate_weighted_l2_norm, global_norm_ptr_physical_coord);

						if (comm->getRank()==0) std::cout << "\n\n\n\nFirst: Global Norm Acceleration: " << global_norm_acceleration << "\n";
						if (comm->getRank()==0) std::cout << "First: Global Norm Physical Coordinates: " << global_norm_physical_coord << "\n\n\n\n\n";
						TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_acceleration > 1e-6, "Acceleration error too large.");
						TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_physical_coord > 1e-15, "Physical coordinate error too large (should be exact).");
					}



					ParticleInsertionTime->start();

					particles->insertParticles(verts_to_insert, 0.0 /*rebuilt halo size*/, false /*repartition*/, false /*insert_physical_coords*/, false /*repartition_with_physical_coords*/);
					ParticleInsertionTime->stop();

					NormTime->stop();
					particles->getFieldManager()->updateFieldsHaloData();
				}

				parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","lagrangian");
				output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "lagrangian_coords_generated_from_lagrangian_coeffs.nc";
				fm.setWriter(output_filename, new_particles);
				fm.write();

				parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","physical");
				output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "physical_coords_generated_from_lagrangian_coeffs.nc";
				fm.setWriter(output_filename, new_particles);
				fm.write();
			}

			// writes out a 1x1x2 shaped rectangle w/extra inserted pts
			output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "lagrange_coords_post_transformation_and_insertion.nc";
			parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","lagrangian");
			fm.setWriter(output_filename, particles);
			fm.write();

			// writes out a 2x2x2 shaped rectangle by expanding in the x and y directions by x and y w/extra inserted pts
			output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "physical_coords_post_transformation_and_insertion.nc";
			parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","physical");
			fm.setWriter(output_filename, particles);
			fm.write();

			// repartition based on physical coordinates
			particles->zoltan2Initialize();

		 	// build halo data (2x as big because of transformation)
			halo_size = 2.0*h_size *
				( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
						+ (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));
			particles->buildHalo(halo_size);

//			{
			// diagnostics
//			particles->resetLagrangianCoords(); // just to see if this fixes things, it didn't
//			particles->resetPhysicalCoords(); // just to see if this fixes things, it didn't
//			particles->zoltan2Initialize();
//			particles->buildHalo(halo_size);
//			}





			if (comm->getRank()==0) std::cout << "Insert test 2 (Inserted as physical, lagrangian derived through interpolation): " << std::endl;

			parameters->get<Teuchos::ParameterList>("remap").set<double>("neighbors needed multiplier", 2.5);
			// has to be enlarged to search double distance in physical coords
			{
			 	// 2. This test creates a new particle set and remaps values onto it from the first set
			 	// It uses the physical coordinates to get coefficients and neighborhoods from which to interpolate
			 	// the Lagrangian coordinates. At the end, it has Lagrangian coordinates that should be interpolated from the first set.
				// Inserted as circle, returned as an ellipse.

				// Particles insertion (Requires interpolation/remap)
				Teuchos::RCP<Compadre::ParticlesT> new_particles =
					Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
				CT* new_coords = (CT*)new_particles->getCoords();
				{
					std::vector<Compadre::XyzVector> verts_to_insert;

					/*
					 * Zoltan2 partition was performed again using physical coordinates
					 *
					 * pmin and pmax are determined using bounding boxes from physical coordinates
					 */
					std::vector<ST> pmin = coords->boundingBoxMinOnProcessor();
					std::vector<ST> pmax = coords->boundingBoxMaxOnProcessor();
					std::cout << comm->getRank() << ": "<< pmin[0] << " " << pmin[1] << " " << pmin[2] << std::endl;
					std::cout << comm->getRank() << ": "<< pmax[0] << " " << pmax[1] << " " << pmax[2] << std::endl;
					Compadre::pool_type pool(1234);
					Compadre::generator_type rand_gen = pool.get_state();

					// we now need to project these points onto the sphere contained in [0,2]x[0,2]x[0,2]

					// make lots of particles and then see which ones fit in the box
					// put points on the sphere centered at (1,1,1) with radius of .75
					Compadre::GlobalConstants consts;
					for (LO i=0; i<NUMBER_OF_PTS_EACH_DIRECTION*NUMBER_OF_PTS_EACH_DIRECTION*comm->getSize(); i++) {
						ST phi = rand_gen.drand(0, 2*consts.Pi());
						ST theta = rand_gen.drand(0, 2*consts.Pi());
						ST radius = .75;
						ST x = radius * sin(theta) * cos(phi);
						ST y = radius * sin(theta) * sin(phi);
						ST z = radius * cos(theta);
						x += 1.0; y += 1.0; z += 1.0; // center at (1,1,1)

						ST eps=1e-2;
						if ((x>pmin[0]+eps && x<pmax[0]-eps) && (y>pmin[1]+eps && y<pmax[1]-eps)	&& (z>pmin[2]+eps && z<pmax[2]-eps))
							verts_to_insert.push_back(Compadre::XyzVector(x, y, z));
					}

					new_coords->insertCoords(verts_to_insert);
					new_particles->resetWithSameCoords();

					Teuchos::RCP<Compadre::RemapManager> rm = Teuchos::rcp(new Compadre::RemapManager(parameters, particles.getRawPtr(), new_particles.getRawPtr(), halo_size));
					Compadre::RemapObject ro(-20); // reserved for interpolating Lagrangian coords from coefficients based on physical coords
					Compadre::RemapObject ra("acceleration");
					rm->add(ro);
					rm->add(ra);

					rm->execute(false /* don't keep neighborhoods */, false /* don't keep GMLS */, false /* don't reuse neighborhoods */, false /* don't reuse GMLS */, true /* use physical coords to form neighborhoods */);

					std::vector<Compadre::XyzVector> verts_screwed_up;
					{
					 	// point by point check to see how far off from correct value
						// check solution
					 	ST acceleration_weighted_l2_norm = 0;
					 	ST physical_coordinate_weighted_l2_norm = 0;
						for( int j =0; j<new_coords->nLocal(); j++){
							Compadre::XyzVector xyz = new_coords->getLocalCoords(j);//, false /*halo*/, false /*use_physical_coords*/); // make sure this matches how they were assigned originally (as a function of Lagrangian coordinates or physical coordinates)

							Compadre::XyzVector computed_xyz = new_coords->getLocalCoords(j, false, false);
							Compadre::XyzVector exact_lagrangian_coord_xyz = xy_inc_fn.evalVector(xyz);
							exact_lagrangian_coord_xyz.x = 0.5*xyz.x;
							exact_lagrangian_coord_xyz.y = 0.5*xyz.y;
							exact_lagrangian_coord_xyz.z = xyz.z;
							std::vector<ST> exact_lagrangian_coord(3);
							std::vector<ST> computed_lagrangian_coord(3);
							exact_lagrangian_coord_xyz.convertToStdVector(exact_lagrangian_coord);
							computed_xyz.convertToStdVector(computed_lagrangian_coord);
							for (LO i=0; i<3; i++) {

								physical_coordinate_weighted_l2_norm += (computed_lagrangian_coord[i] - exact_lagrangian_coord[i])*(computed_lagrangian_coord[i] - exact_lagrangian_coord[i]);
								if ( (computed_lagrangian_coord[i] - exact_lagrangian_coord[i])*(computed_lagrangian_coord[i] - exact_lagrangian_coord[i])  > 1e-7) {
									// note computed always less than physical
									std::cout << j << "," << i << " " << computed_lagrangian_coord[i] << " " << exact_lagrangian_coord[i] << std::endl;
									verts_screwed_up.push_back(Compadre::XyzVector(xyz.x, xyz.y, xyz.z));
								}
							}

							std::vector<ST> computed_acceleration = new_particles->getFieldManagerConst()->getFieldByName("acceleration")->getLocalVectorVal(j);
							Compadre::XyzVector exact_acceleration_xyz = sinx_sol.evalVector(exact_lagrangian_coord_xyz);
							std::vector<ST> exact_acceleration(3);
							exact_acceleration_xyz.convertToStdVector(exact_acceleration);
							for (LO i=0; i<3; i++) {
								acceleration_weighted_l2_norm += (computed_acceleration[i] - exact_acceleration[i])*(computed_acceleration[i] - exact_acceleration[i]);
							}

						}

//						{
//						// diagnostic
//						// won't run if nothing is messed up
//							// temporary for diagnostic
//							Teuchos::RCP<Compadre::ParticlesT> new_particles2 =
//								Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
//							CT* new_coords2 = (CT*)new_particles2->getCoords();
//							if (parameters->get<std::string>("simulation type")=="lagrangian") {
//								new_coords2->insertCoords(verts_screwed_up);
//							}
//							new_particles2->resetWithSameCoords();
//
//							parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","lagrangian");
//							output_filename = "screwed_up_lagrangian.nc";
//							fm.setWriter(output_filename, new_particles2);
//							fm.write();
//						}

						acceleration_weighted_l2_norm = sqrt(acceleration_weighted_l2_norm) /((ST)new_coords->nGlobalMax());
						physical_coordinate_weighted_l2_norm = sqrt(physical_coordinate_weighted_l2_norm) /((ST)new_coords->nGlobalMax());

						ST global_norm_acceleration, global_norm_physical_coord;

						Teuchos::Ptr<ST> global_norm_ptr_acceleration(&global_norm_acceleration);
						Teuchos::Ptr<ST> global_norm_ptr_physical_coord(&global_norm_physical_coord);

						Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, acceleration_weighted_l2_norm, global_norm_ptr_acceleration);
						Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, physical_coordinate_weighted_l2_norm, global_norm_ptr_physical_coord);

						if (comm->getRank()==0) std::cout << "\n\n\n\nSecond: Global Norm Acceleration: " << global_norm_acceleration << "\n";
						if (comm->getRank()==0) std::cout << "Second: Global Norm Lagrangian Coordinates: " << global_norm_physical_coord << "\n\n\n\n\n";
						TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_acceleration > 5e-5, "Acceleration error too large.");
						TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_physical_coord > 1e-15, "Physical coordinate error too large (should be exact).");
					}

					NormTime->stop();
					MiscTime->start();
					ParticleInsertionTime->start();

					particles->insertParticles(verts_to_insert);
//					Part->start();
					ParticleInsertionTime->stop();
					particles->getFieldManager()->updateFieldsHaloData();
				}

				// should look like an ellipse centered at (.5,.5,1)
				parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","lagrangian");
				output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "lagrangian_coords_generated_from_physical_coeffs.nc";
				fm.setWriter(output_filename, new_particles);
				fm.write();

				// should look like a sphere centered at (1,1,1)
				parameters->get<Teuchos::ParameterList>("io").set<std::string>("coordinate type","physical");
				output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "physical_coords_generated_from_physical_coeffs.nc";
				fm.setWriter(output_filename, new_particles);
				fm.write();
			}



		 	// point by point check to see how far off from correct value
			// check solution
		 	ST acceleration_weighted_l2_norm = 0;
		 	ST velocity_weighted_l2_norm = 0;
		 	ST pressure_weighted_l2_norm = 0;
		 	ST physical_coordinate_weighted_l2_norm = 0;
			for( int j =0; j<coords->nLocal(); j++){
				Compadre::XyzVector xyz = coords->getLocalCoords(j, false /*halo*/, false /*use_physical_coords*/); // make sure this matches how they were assigned originally (as a function of Lagrangian coordinates or physical coordinates)

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

				Compadre::XyzVector exact_physical_coord_xyz = xy_inc_fn.evalVector(xyz);
				exact_physical_coord_xyz.x += xyz.x;
				exact_physical_coord_xyz.y += xyz.y;
				exact_physical_coord_xyz.z += xyz.z;
				std::vector<ST> exact_physical_coord(3);
				Compadre::XyzVector computed_xyz = coords->getLocalCoords(j); // modified to false
				std::vector<ST> computed_physical_coord(3);
				exact_physical_coord_xyz.convertToStdVector(exact_physical_coord);
				computed_xyz.convertToStdVector(computed_physical_coord);
				for (LO i=0; i<3; i++) {
					physical_coordinate_weighted_l2_norm += (computed_physical_coord[i] - exact_physical_coord[i])*(computed_physical_coord[i] - exact_physical_coord[i]);
				}
			}
			acceleration_weighted_l2_norm = sqrt(acceleration_weighted_l2_norm) /((ST)coords->nGlobalMax());
			velocity_weighted_l2_norm = sqrt(velocity_weighted_l2_norm) /((ST)coords->nGlobalMax());
			pressure_weighted_l2_norm = sqrt(pressure_weighted_l2_norm) /((ST)coords->nGlobalMax());
			physical_coordinate_weighted_l2_norm = sqrt(physical_coordinate_weighted_l2_norm) /((ST)coords->nGlobalMax());

			ST global_norm_acceleration, global_norm_velocity, global_norm_pressure, global_norm_physical_coord;

			Teuchos::Ptr<ST> global_norm_ptr_acceleration(&global_norm_acceleration);
			Teuchos::Ptr<ST> global_norm_ptr_velocity(&global_norm_velocity);
			Teuchos::Ptr<ST> global_norm_ptr_pressure(&global_norm_pressure);
			Teuchos::Ptr<ST> global_norm_ptr_physical_coord(&global_norm_physical_coord);

			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, acceleration_weighted_l2_norm, global_norm_ptr_acceleration);
			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, velocity_weighted_l2_norm, global_norm_ptr_velocity);
			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, pressure_weighted_l2_norm, global_norm_ptr_pressure);
			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, physical_coordinate_weighted_l2_norm, global_norm_ptr_physical_coord);

			if (comm->getRank()==0) std::cout << "\n\n\n\nGlobal Norm Acceleration: " << global_norm_acceleration << "\n";
			if (comm->getRank()==0) std::cout << "Global Norm Velocity: " << global_norm_velocity << "\n";
			if (comm->getRank()==0) std::cout << "Global Norm Pressure: " << global_norm_pressure << "\n";
			if (comm->getRank()==0) std::cout << "Global Norm Physical Coordinates: " << global_norm_physical_coord << "\n\n\n\n\n";

			output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "lagrangian_particle_output.nc";
			fm.setWriter(output_filename, particles);
			fm.write();

			TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_acceleration > 1e-6, "Acceleration error too large.");
			TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_velocity > 1e-15, "Velocity error too large (should be exact).");
			TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_pressure > 1e-15, "Pressure error too large (should be exact).");
			TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_physical_coord > 1e-15, "Physical coordinate error too large (should be exact).");
			NormTime->stop();
		 }
		MiscTime->stop();
//		Part->stop();
	}
	if (comm->getRank()==0) parameters->print();
	Teuchos::TimeMonitor::summarize();
	Kokkos::finalize();
return 0;
}

