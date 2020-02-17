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

typedef int LO;
typedef long GO;
typedef double ST;

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
	Teuchos::RCP<Teuchos::Time> NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Neighbor Search Time");

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

	const LO my_coloring = parameters->get<LO>("my coloring");
	const LO peer_coloring = parameters->get<LO>("peer coloring");

	Teuchos::RCP<const Teuchos::Comm<int> > comm = global_comm->split(my_coloring, global_comm->getRank());


	std::cout << "Talking from global proc: " << global_comm->getRank() << " local proc: " << comm->getRank() << " of " << global_comm->getSize() << " " << comm->getSize() << std::endl;

	LO NUMBER_OF_PTS_EACH_DIRECTION;
	if (my_coloring==25) {
		NUMBER_OF_PTS_EACH_DIRECTION = 21;
	} else {
		NUMBER_OF_PTS_EACH_DIRECTION = 27;
	}

	{
		MiscTime->start();
		{
			//
			// Remote Data Test (Two differing distributions of particles)
			//

			typedef Compadre::EuclideanCoordsT CT;
		 	Teuchos::RCP<Compadre::ParticlesT> particles =
		 		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
		 	CT* coords = (CT*)particles->getCoords();

			ST h_size = 1.0/((ST)NUMBER_OF_PTS_EACH_DIRECTION-1);

		 	// Coordinate insertion
		 	{
				std::vector<Compadre::XyzVector> verts_to_insert;
				if (my_coloring==25) {
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
				} else {
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
					// these pts will not find a match on the other side
					// for testing what happens when there is empty processor box intersections
					if (comm->getRank()==2) {
						for (LO i=0; i<NUMBER_OF_PTS_EACH_DIRECTION; i++) {
							for (LO j=0; j<NUMBER_OF_PTS_EACH_DIRECTION; j++) {
								for (LO k=0; k<NUMBER_OF_PTS_EACH_DIRECTION-1; k++) {
									verts_to_insert.push_back(Compadre::XyzVector(h_size*i, h_size*j, 2+h_size*(k+1)));
								}
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

			Teuchos::RCP<Compadre::RemoteDataManager> remoteDataManager = Teuchos::rcp(new Compadre::RemoteDataManager(global_comm, comm, coords, my_coloring, peer_coloring));
		 	Teuchos::RCP<Compadre::ParticlesT> peer_processors_particles =
		 		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
			remoteDataManager->putRemoteCoordinatesInParticleSet(peer_processors_particles.getRawPtr());

			Compadre::FileManager fm;
			std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "after_swap" + std::to_string(my_coloring) + ".nc";
			fm.setWriter(output_filename, peer_processors_particles);
			fm.write();
		}
		MiscTime->stop();
	}


	Teuchos::TimeMonitor::summarize();
	Kokkos::finalize();
return 0;
}

