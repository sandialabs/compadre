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
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_VTKInformation.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <GMLS_Config.h>

#include <GMLS.hpp>

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::CoordsT coords_type;
typedef Compadre::XyzVector xyz_type;

int main (int argc, char* args[]) {

	std::string current_path = args[1];
	int _type = std::stoi(args[2]); // 0 for point, 1 for grad_x, 2 for grad_y
	//printf("name was: %s\n", current_path.c_str());

	Teuchos::RCP<Compadre::ParameterManager> parameter_manager;
	parameter_manager = Teuchos::rcp(new Compadre::ParameterManager());

	if (parameter_manager->helpRequested()) return 0;
	if (parameter_manager->parseError()) return -1;

	Teuchos::RCP<Teuchos::ParameterList> parameters = parameter_manager->getList();

	Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::oblackholestream bstream;
	Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

	Kokkos::initialize(argc, args);
	
	const int procRank = comm->getRank();
	const int nProcs = comm->getSize();

	Teuchos::RCP<Teuchos::Time> MiscTime = Teuchos::TimeMonitor::getNewCounter ("Miscellaneous");
	Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");
	Teuchos::RCP<Teuchos::Time> CoordinateInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Coordinate Insertion Time");
	Teuchos::RCP<Teuchos::Time> ParticleInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Particle Insertion Time");
	Teuchos::RCP<Teuchos::Time> NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Neighbor Search Time");

	parameters->get<Teuchos::ParameterList>("remap").set<int>("porder",2);
	parameters->get<Teuchos::ParameterList>("remap").set<double>("neighbors needed multiplier",1.5);
	parameters->get<Teuchos::ParameterList>("remap").set<double>("cutoff multiplier", 1.1); // must increase with porder
	parameters->get<Teuchos::ParameterList>("remap").set<double>("epsilon multiplier", 1.1); // must increase with porder
	// There is a balance between "neighbors needed multiplier" and "cutoff multiplier"
	// Their product must increase somewhat with "porder", but "neighbors needed multiplier"
	// requires more of an increment than "cutoff multiplier" to achieve the required threshold (the other staying the same)
	// parameters->get<Teuchos::ParameterList>("neighborhood").set<bool>("search: dynamic", true);
	// parameters->get<Teuchos::ParameterList>("neighborhood").set<double>("search: multiplier", 0.125);
	// parameters->get<Teuchos::ParameterList>("neighborhood").set<double>("search: size", 0.2);
	// parameters->get<Teuchos::ParameterList>("neighborhood").set<double>("halo: multiplier", 1.3);
	// parameters->get<Teuchos::ParameterList>("neighborhood").set<double>("halo: size", 0.2);

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

	                //std::ifstream file(current_path+"/lowerpoints.txt", std::ios::in);
	                std::ifstream file(current_path+"/lowerpoints.txt");
	                int numRow;
	                file >> numRow;
	
			ST h_size;
			std::vector<Compadre::XyzVector> verts_to_insert;
			{
	                	std::vector<std::vector<double> > allData(numRow, std::vector<double>(3,0));
	                	for (int i=0; i<numRow; i++) {
	                	  file >> allData[i][0];
	                	  file >> allData[i][1];
                        	  verts_to_insert.push_back(Compadre::XyzVector(allData[i][0], allData[i][1], 0));
	                	  //std::cout << allData[i][0] << " " << allData[i][1] << " " << allData[i][2] << std::endl;
	                	}

				h_size = 1000*(std::sqrt(std::pow(allData[0][0]-allData[1][0],2)+std::pow(allData[0][1]-allData[1][1],2)));
			}
			MiscTime->stop();
			CoordinateInsertionTime->start();
			coords->insertCoords(verts_to_insert);
			CoordinateInsertionTime->stop();
			MiscTime->start();

		 	particles->resetWithSameCoords(); // must be called because particles doesn't know about coordinate insertions



			//{
			//	particles->getFieldManager()->createField(1,"id","none");
			//	Compadre::device_view_type field_vals = particles->getFieldManager()->getFieldByName("id")->getDeviceView();
			//	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,allData.size()), KOKKOS_LAMBDA(const int i) {
			//		field_vals(i,0) = i;
			//	});
			//}




		 	// build halo data
			ST halo_size = h_size *
					( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
							+ (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));

			particles->zoltan2Initialize();
			particles->buildHalo(halo_size);


			//std::string output_filename = "remap_coords_output.nc";
			//Compadre::FileManager fm;
			//fm.setWriter(output_filename, particles);
			//fm.write();

			//// register fields
			//particles->getFieldManager()->createField(1,"val","none");
			//Compadre::device_view_type field_vals = particles->getFieldManager()->getFieldByName("val")->getDeviceView();
			//Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,allData.size()), KOKKOS_LAMBDA(const int i) {
			//	field_vals(i,0) = allData[i][2];
			//});

			//particles->getFieldManager()->updateFieldsHaloData();
			//particles->createNeighborhood();

		 	Teuchos::RCP<Compadre::ParticlesT> new_particles =
		 		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
		 	CT* new_coords = (CT*)new_particles->getCoords();

			{
				//std::ifstream file("./higherpoints.txt", std::ios::in);
				std::ifstream file(current_path+"/higherpoints.txt");
				int numCell;
				file >> numCell;
				
				std::vector<std::vector<double> > allData(numCell, std::vector<double>(3,0));
				std::vector<Compadre::XyzVector> verts_to_insert;
	                	for (int i=0; i<numCell; i++) {
	                	  file >> allData[i][0];
	                	  file >> allData[i][1];
                        	  verts_to_insert.push_back(Compadre::XyzVector(allData[i][0], allData[i][1], 0));
	                	}

				new_coords->insertCoords(verts_to_insert);
				new_particles->resetWithSameCoords();
				//new_particles->zoltan2Initialize();
				//new_particles->buildHalo(halo_size);

				std::string method = parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("method");
				transform(method.begin(), method.end(), method.begin(), ::tolower);


				typedef Compadre::NeighborhoodT neighbors_type;
				typedef Compadre::NanoFlannInformation nanoflann_neighbors_type;
				Teuchos::RCP<neighbors_type> neighborhoodInfo;
					LO maxLeaf = parameters->get<Teuchos::ParameterList>("neighborhood").get<int>("max leaf");
					neighborhoodInfo = Teuchos::rcp_static_cast<neighbors_type>(Teuchos::rcp(
							new nanoflann_neighbors_type(particles.getRawPtr(), parameters, maxLeaf, new_particles.getRawPtr())));

				LO neighbors_needed;
				neighbors_needed = Compadre::GMLS::getNP(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),2);
		
				LO extra_neighbors = parameters->get<Teuchos::ParameterList>("remap").get<double>("neighbors needed multiplier") * neighbors_needed;
		
				neighborhoodInfo->setAllHSupportSizes(parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"));
				neighborhoodInfo->constructAllNeighborList(halo_size, extra_neighbors);

				LO max_neighbors = neighborhoodInfo->getMaxNumNeighbors();
				std::cout << "Max neighbors: " << max_neighbors << std::endl;
		
				// generate the interpolation operator and call the coefficients needed (storing them)
				const coords_type* target_coords = new_particles->getCoordsConst();
				const coords_type* source_coords = particles->getCoordsConst();
		
				//std::vector<std::vector<ST> > alphas(new_particles->getCoordsConst()->nLocal());
				//std::vector<ST> alphas(new_particles->getCoordsConst()->nLocal());
				//std::ofstream out_file("./out.txt", std::ios::out);
				std::ofstream out_file(current_path+"/out.txt");
				out_file.precision(16);
                                out_file << numCell << std::endl;
				//for (LO i=0; i<new_particles->getCoordsConst()->nLocal(); i++) {


				//****************
				//
				//  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
				//
				//****************

				const std::vector<std::vector<std::pair<size_t, ST> > >& all_neighbors = neighborhoodInfo->getAllNeighbors();

				size_t max_num_neighbors = 0;
				Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()),
						KOKKOS_LAMBDA (const int i, size_t &myVal) {
					myVal = (all_neighbors[i].size() > myVal) ? all_neighbors[i].size() : myVal;
				}, Kokkos::Experimental::Max<size_t>(max_num_neighbors));

				Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), max_num_neighbors+1);
				Kokkos::View<int**>::HostMirror kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);

				// fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
					const int num_i_neighbors = all_neighbors[i].size();
					for (int j=1; j<num_i_neighbors+1; ++j) {
						kokkos_neighbor_lists_host(i,j) = all_neighbors[i][j-1].first;
					}
					kokkos_neighbor_lists_host(i,0) = num_i_neighbors;
				});

				Kokkos::View<double**> kokkos_augmented_source_coordinates("source_coordinates", source_coords->nLocal(true /* include halo in count */), source_coords->nDim());
				Kokkos::View<double**>::HostMirror kokkos_augmented_source_coordinates_host = Kokkos::create_mirror_view(kokkos_augmented_source_coordinates);

				// fill in the source coords, adding regular with halo coordiantes into a kokkos view
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int i) {
					xyz_type coordinate = source_coords->getLocalCoords(i, true /*include halo*/, true);
					kokkos_augmented_source_coordinates_host(i,0) = coordinate.x;
					kokkos_augmented_source_coordinates_host(i,1) = coordinate.y;
					kokkos_augmented_source_coordinates_host(i,2) = coordinate.z;
				});

				Kokkos::View<double**> kokkos_target_coordinates("target_coordinates", target_coords->nLocal(), target_coords->nDim());
				Kokkos::View<double**>::HostMirror kokkos_target_coordinates_host = Kokkos::create_mirror_view(kokkos_target_coordinates);
				// fill in the target, adding regular coordiantes only into a kokkos view
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
					xyz_type coordinate = target_coords->getLocalCoords(i, false /*include halo*/, true);
					kokkos_target_coordinates_host(i,0) = coordinate.x;
					kokkos_target_coordinates_host(i,1) = coordinate.y;
					kokkos_target_coordinates_host(i,2) = coordinate.z;
				});

				auto epsilons = neighborhoodInfo->getHSupportSizes()->getLocalView<const Compadre::host_view_type>();
				Kokkos::View<double*> kokkos_epsilons("target_coordinates", target_coords->nLocal(), target_coords->nDim());
				Kokkos::View<double*>::HostMirror kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
					kokkos_epsilons_host(i) = epsilons(i,0);
				});

				//****************
				//
				//  End of data copying
				//
				//****************

				Teuchos::RCP<GMLS> _GMLS = Teuchos::rcp(new GMLS(kokkos_neighbor_lists_host,
							kokkos_augmented_source_coordinates_host,
							kokkos_target_coordinates,
							kokkos_epsilons_host,
							parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
							"SVD",
							parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder")));

				_GMLS->setWeightingType(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
				_GMLS->setWeightingPower(parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));

				std::vector<TargetOperation> lro(2);
				lro[0]=TargetOperation::ScalarPointEvaluation;
				lro[1]=TargetOperation::GradientOfScalarPointEvaluation;
    				_GMLS->addTargets(lro, 3 /* dimension */);
				_GMLS->generateAlphas(); // all operations requested

				for (LO i=0; i<new_particles->getCoordsConst()->nLocal(); i++) {
					const std::vector<std::pair<size_t, ST> > neighbors = neighborhoodInfo->getNeighbors(i);
					const LO num_neighbors = neighbors.size();
					out_file << num_neighbors << " ";
					for (LO j=0; j<num_neighbors; j++) {
                                                //out_file << field_vals(neighbors[j].first,0) << " ";
                                                out_file << neighbors[j].first << " ";
					}
                                	for (int j=0; j<num_neighbors; j++) {
                                	//for (int j=0; j<alphas.size(); j++) {
						if (_type == 0) {
							out_file << _GMLS->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j) << " ";
						} else if (_type == 1) {
							out_file << _GMLS->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, j) << " ";
						} else if (_type == 2) {
							out_file << _GMLS->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, j) << " ";
						}
					}
					out_file << std::endl;
				}
 				{
					std::ofstream out_file2(current_path+"/complete.txt", std::ios::out);
                                	out_file2 << 1 << std::endl;;
				}
			}
		 }
		MiscTime->stop();
	}
	//if (comm->getRank()==0) parameters->print();
	Teuchos::TimeMonitor::summarize();
	Kokkos::finalize();
return 0;
}

