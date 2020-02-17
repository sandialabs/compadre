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
#include <Compadre_RemoteDataManager.hpp>

#include <iostream>

using namespace Compadre;

typedef int LO;
typedef long GO;
typedef double ST;
typedef XyzVector xyz_type;

int main (int argc, char* args[]) {

	Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::oblackholestream bstream;
	Teuchos::RCP<const Teuchos::Comm<int> > global_comm = Teuchos::DefaultComm<int>::getComm();

	Kokkos::initialize(argc, args);

	Teuchos::RCP<Teuchos::Time> ParameterTime = Teuchos::TimeMonitor::getNewCounter ("Parameter Initialization");
	Teuchos::RCP<Teuchos::Time> MiscTime = Teuchos::TimeMonitor::getNewCounter ("Miscellaneous");
	Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");
	Teuchos::RCP<Teuchos::Time> CoordinateInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Coordinate Insertion Time");
	Teuchos::RCP<Teuchos::Time> ParticleInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Particle Insertion Time");
	Teuchos::RCP<Teuchos::Time> NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Canga - Neighbor Search Time");
	Teuchos::RCP<Teuchos::Time> ReadTime = Teuchos::TimeMonitor::getNewCounter ("Read Time");

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

#ifndef COMPADREHARNESS_USE_COMPOSE
    if (parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm")=="CAAS") {
        printf("\n\n\n\n\n\n\n\nCompose package called but not built. Enable with '-D CompadreHarness_USE_Compose:BOOL=ON'.\n\n\n\n\n\n\n\n");
        // special return code for testing
        return 77;
    }
#endif

	const LO my_coloring = parameters->get<LO>("my coloring");
	const LO peer_coloring = parameters->get<LO>("peer coloring");

	Teuchos::RCP<const Teuchos::Comm<int> > comm = global_comm->split(my_coloring, global_comm->getRank());



	std::cout << "Talking from global proc: " << global_comm->getRank() << " local proc: " << comm->getRank() << " of " << global_comm->getSize() << " " << comm->getSize() << std::endl;

	{

		// Should read in two different meshes (based on coloring of processes)
		//
		//  - point data as one cloud
		//  - cell data as another cloud with centroids as points
		//  - edge data as another cloud with edge centers as points
		//
		// Should be able to:
		//
		//  - Define a field at the points in question
		//  - Be able to choose a subset based on their flag (done)
		//  - Only perform cell finding with the appropriate particles (create a new particle set from the previous one)
		//  - Swap / reconstruct field from other program
		//  - Swap / send back new information
		//  - Check the true solution against the computed solution from the other side (only at flagged points)
		//

		MiscTime->start();
		{
			//
			// Remote Data Test (Two differing distributions of particles)
			//

			typedef Compadre::EuclideanCoordsT CT;
		 	Teuchos::RCP<Compadre::ParticlesT> particles =
		 		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
		 	CT* coords = (CT*)particles->getCoords();

		 	// Read in appropriate file here

			//Read in data file.
			ReadTime->start();
			Compadre::FileManager fm;
			std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));
			fm.setReader(testfilename, particles, parameters->get<Teuchos::ParameterList>("io").get<std::string>("nc type"));
            fm.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
            fm.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
            fm.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
			fm.read();
			ReadTime->stop();


			// define some field on the source particles
			// define a field specific to the process running
			if (my_coloring == 25) {
				Compadre::ShallowWaterTestCases shallow_water_case_2_sol(2);
				particles->getFieldManager()->createField(3,"source_sinx","m^2/1a");
				particles->getFieldManager()->getFieldByName("source_sinx")->
						localInitFromVectorFunction(&shallow_water_case_2_sol);
				Compadre::SphereHarmonic sphere_harmonic_sol(2,2);
				particles->getFieldManager()->createField(1,"source_sphere_harmonics","m^2/1b");
				particles->getFieldManager()->getFieldByName("source_sphere_harmonics")->
						localInitFromScalarFunction(&sphere_harmonic_sol);

                // this field will not remap accurately as it is a vector that is not
                // tangent to the manifold
				Compadre::ConstantEachDimension constant_sol(1,2,3);
				particles->getFieldManager()->createField(3,"source_constant","m^2/1c");
				particles->getFieldManager()->getFieldByName("source_constant")->
						localInitFromVectorFunction(&constant_sol);
			} else if (my_coloring == 33) {
                Compadre::SphereTestVelocity sphere_velocity_sol;
				particles->getFieldManager()->createField(3,"source_x2","m^2-2a--");
				particles->getFieldManager()->getFieldByName("source_x2")->
						localInitFromVectorFunction(&sphere_velocity_sol);
			}

		 	// calculate h_size for mesh

		 	// build halo data
			ST halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size") *
					( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
							+ (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));
			particles->zoltan2Initialize();
			particles->buildHalo(halo_size);
			particles->getFieldManager()->updateFieldsHaloData();

			std::string output_filename1 = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "no_swap" + std::to_string(my_coloring) + ".nc";
			fm.setWriter(output_filename1, particles);
			fm.write();


//			std::vector<int> flags_to_transfer(2);
//			flags_to_transfer[0] = 1;
//			flags_to_transfer[1] = 2;
//			// should add in a filter in the remoteDataManager
//			Compadre::host_view_type flags = particles->getFlags()->getLocalView<Compadre::host_view_type>();
//			Teuchos::RCP<Compadre::RemoteDataManager> remoteDataManager =
//					Teuchos::rcp(new Compadre::RemoteDataManager(global_comm, comm, coords, my_coloring, peer_coloring, flags, flags_to_transfer));


			Teuchos::RCP<Compadre::RemoteDataManager> remoteDataManager =
					Teuchos::rcp(new Compadre::RemoteDataManager(global_comm, comm, coords, my_coloring, peer_coloring));

		 	Teuchos::RCP<Compadre::ParticlesT> peer_processors_particles =
		 		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));

		 	remoteDataManager->putRemoteCoordinatesInParticleSet(peer_processors_particles.getRawPtr());

            if (parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm")!="NONE") {
                std::string target_weights_name = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("target weighting field name");
                std::string source_weights_name = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("source weighting field name");
                TEUCHOS_TEST_FOR_EXCEPT_MSG(target_weights_name=="", "\"target weighting field name\" not specified in parameter list.");
                TEUCHOS_TEST_FOR_EXCEPT_MSG(source_weights_name=="", "\"source weighting field name\" not specified in parameter list.");
                remoteDataManager->putRemoteWeightsInParticleSet(particles.getRawPtr(), peer_processors_particles.getRawPtr(), target_weights_name);
            }

		 	std::vector<Compadre::RemapObject> remap_vec;
			if (my_coloring == 25) {
//				Compadre::RemapObject r1("source_x2", "peer_x2");
				Compadre::RemapObject r1("source_x2", "peer", VectorPointEvaluation, VectorOfScalarClonesTaylorPolynomial, ManifoldVectorPointSample);
                Compadre::OptimizationObject opt_obj = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, parameters->get<Teuchos::ParameterList>("remap").get<double>("global lower bound")/*global lower bound*/, parameters->get<Teuchos::ParameterList>("remap").get<double>("global upper bound")/*global upper bound*/); 
                r1.setOptimizationObject(opt_obj);
				remap_vec.push_back(r1);
			} else if (my_coloring == 33) {
				Compadre::RemapObject r1("source_constant", "peer_constant");
				Compadre::RemapObject r2("source_sinx", "peer");
				remap_vec.push_back(r1);
				remap_vec.push_back(r2);
                Compadre::OptimizationObject opt_obj = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, parameters->get<Teuchos::ParameterList>("remap").get<double>("global lower bound")/*global lower bound*/, parameters->get<Teuchos::ParameterList>("remap").get<double>("global upper bound")/*global upper bound*/); 
				Compadre::RemapObject r3("source_sphere_harmonics", "peer_harmonics", ScalarPointEvaluation, ScalarTaylorPolynomial, PointSample);
				remap_vec.push_back(r3);
			}
		 	remoteDataManager->remapData(remap_vec, parameters, particles.getRawPtr(), peer_processors_particles.getRawPtr(), halo_size);

			std::string output_filename2 = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "before_swap" + std::to_string(my_coloring) + ".nc";
			fm.setWriter(output_filename2, peer_processors_particles);
			fm.write();

			std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "after_swap" + std::to_string(my_coloring) + ".nc";
			fm.setWriter(output_filename, particles);
			fm.write();

//			output_filename = "after_swap" + std::to_string(my_coloring) + ".nc";
//			fm.setWriter(output_filename, particles);
//			fm.write();

		 	// compare errors against
		 	// get appropriate "true solution" for the peer processes
			Teuchos::RCP<Compadre::AnalyticFunction> function;
			if (my_coloring == 25) {
				function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SphereTestVelocity));
			} else if (my_coloring == 33) {
				function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ShallowWaterTestCases(2)));

                // this field will not remap accurately as it is a vector that is not
                // tangent to the manifold
//				function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ConstantEachDimension(1,2,3)));
			}

			// check solution
			double norm = 0.0;
			xyz_type exact = 0.0;

			for( int j=0; j<coords->nLocal(); j++){
				xyz_type xyz = coords->getLocalCoords(j);
				exact = function->evalVector(xyz);
				std::vector<ST> vals = particles->getFieldManager()->getFieldByName("peer")->getLocalVectorVal(j);
				norm += (exact.x - vals[0])*(exact.x-vals[0]);
				norm += (exact.y - vals[1])*(exact.y-vals[1]);
				norm += (exact.z - vals[2])*(exact.z-vals[2]);
			}
			norm /= (double)(coords->nGlobalMax());

			ST global_norm = 0;
			Teuchos::Ptr<ST> global_norm_ptr(&global_norm);
			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, norm, global_norm_ptr);
			global_norm = sqrt(global_norm);
			if (my_coloring == 25) {
				if (comm->getRank()==0) std::cout << "Global Norm of Spherical Velocity: " << global_norm << "\n";
			} else if (my_coloring == 33) {
				if (comm->getRank()==0) std::cout << "Global Norm of Shallow Water Test Case 2: " << global_norm << "\n";
			}

		}
		MiscTime->stop();
	}


	Teuchos::TimeMonitor::summarize();
	Kokkos::finalize();
return 0;
}

