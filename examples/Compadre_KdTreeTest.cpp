#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include "Compadre_EuclideanCoordsT.hpp"
#include "Compadre_SphericalCoordsT.hpp"
#include "Compadre_AnalyticFunctions.hpp"
#include <TPL/nanoflann/nanoflann.hpp>
#include "Compadre_nanoflannPointCloudT.hpp"

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::EuclideanCoordsT coord_type;
typedef Compadre::XyzVector xyz_type;

template <typename scalar_type, typename local_index_type, typename global_index_type>
local_index_type localKDtreeDemo(const global_index_type N, coord_type* coords,
								   const xyz_type& queryPt) {
	const int nDim = 3;
	
	typedef Compadre::LocalPointCloudT cloud_type;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
				nanoflann::L2_Simple_Adaptor<scalar_type, cloud_type>, cloud_type, nDim> tree_type;
	
	cloud_type cloud(coords, true /*use_physical_coords_if_lagrangian*/, false /* bbox not precomputed */);
	scalar_type query_arr[3] = {queryPt.x, queryPt.y, queryPt.z};
	
	const local_index_type maxLeaf = 10;
	
	tree_type index(nDim, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf));
	index.buildIndex();

	// knn search
	const local_index_type num_results = 1;
	local_index_type return_index;
	scalar_type sq_dist_out;
	nanoflann::KNNResultSet<scalar_type, local_index_type, global_index_type> resultSet(num_results);
	resultSet.init(&return_index, &sq_dist_out);
	index.findNeighbors(resultSet, &query_arr[0], nanoflann::SearchParams(maxLeaf));
	
	std::cout << "\tlocal knnSearch : return_index = " << return_index << ", output dist. squared = " << sq_dist_out << std::endl;
	return return_index;
}

template <typename scalar_type, typename local_index_type, typename global_index_type>
global_index_type globalKDTreeDemo(const global_index_type N, coord_type* coords,
	const xyz_type& queryPt) {
	
	const int nDim = 3;
	
	typedef Compadre::GlobalPointCloudT cloud_type;
	typedef nanoflann::KDTreeSingleIndexAdaptor<
				nanoflann::L2_Simple_Adaptor<scalar_type, cloud_type>, cloud_type, nDim> tree_type;
	cloud_type cloud(coords);
	
	scalar_type query_arr[3] = {queryPt.x, queryPt.y, queryPt.z};
	
	const local_index_type maxLeaf = 10;
	
	tree_type index(nDim, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf));
	index.buildIndex();
	
	// knn search
	const local_index_type num_results = 1;
	global_index_type return_index;
	scalar_type sq_dist_out;
	nanoflann::KNNResultSet<scalar_type, global_index_type, global_index_type> resultSet(num_results);
	resultSet.init(&return_index, &sq_dist_out);
	index.findNeighbors(resultSet, &query_arr[0], nanoflann::SearchParams(maxLeaf));
	
	std::cout << "\tglobal knnSearch : return_index = " << return_index << ", output dist. squared = " << sq_dist_out << std::endl;
	return return_index;
}

struct KokkosHello{
	const int rank;
	
	KokkosHello(const int _rank) : rank(_rank) {};
	
	KOKKOS_INLINE_FUNCTION
	void operator() (const int i) const {
		std::printf("rank %d, i = %d\n", rank, i);
	}
};

int main(int argc, char* args[]) {
	Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
	Kokkos::initialize(argc, args);
	
	const int procRank = comm->getRank();
	const int nProcs = comm->getSize();
	
// 	std::cout << "Hello from MPI rank " << procRank << ", Kokkos execution space " 
// 			  << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
// 			  
// 	Kokkos::parallel_for(nProcs, KokkosHello(procRank));
	{
	const LO n = 20;
	const GO n3 = n * n * n;
	Teuchos::RCP<coord_type> eucCoords = Teuchos::rcp(new coord_type(n3, comm));
	eucCoords->initRandom(2.0, procRank);
	eucCoords->zoltan2Init();
	eucCoords->zoltan2Partition();

	// buildHalo(double tol) can be tested by setting it larger than the domain, in which case
	// each processor will find the closest point globally, or setting it to zero, in which 
	// case only one processor can find the closest point globally in the local search

	// each processor will find its own unique minimizer (empty halo)
	eucCoords->buildHalo(0.0);
	
	const xyz_type qVec(0.5, 0.5, 0.5);
	
	{
		const LO nearInd = localKDtreeDemo<ST, LO, GO>(n, eucCoords.getRawPtr(), qVec);
		const xyz_type nearVec = eucCoords->getLocalCoords(nearInd, true /*use_halo*/);
		const ST dist = eucCoords->distance(nearInd, qVec);
		for (int i = 0; i < nProcs; ++i) {
			comm->barrier();
			if (procRank == i ) {	
				std::cout << "Proc " << procRank << " localKnnSearch :" << std::endl;
				std::cout << "\tnearest particle to " << qVec << " is at " << nearVec << std::endl;
				std::cout << "\tdist = " << dist << std::endl;
				std::cout << "\tdist. sq. = " << dist * dist << std::endl;
			}
		}
		comm->barrier();
		std::cout << "LOCAL DEMO COMPLETE " << procRank << std::endl;
	}

	// each processor will find the global minimizer (halo is the complement of the local points)
	eucCoords->buildHalo(50.0);

	{
		const LO nearInd = localKDtreeDemo<ST, LO, GO>(n, eucCoords.getRawPtr(), qVec);
		const xyz_type nearVec = eucCoords->getLocalCoords(nearInd, true /*use_halo*/);
		const ST dist = eucCoords->distance(nearInd, qVec);
		for (int i = 0; i < nProcs; ++i) {
			comm->barrier();
			if (procRank == i ) {	
				std::cout << "Proc " << procRank << " localKnnSearch :" << std::endl;
				std::cout << "\tnearest particle to " << qVec << " is at " << nearVec << std::endl;
				std::cout << "\tdist = " << dist << std::endl;
				std::cout << "\tdist. sq. = " << dist * dist << std::endl;
			}
		}
		comm->barrier();
		std::cout << "LOCAL DEMO COMPLETE " << procRank << std::endl;
	}
	
	{
		const GO nearInd = globalKDTreeDemo<ST, LO, GO>(n3, eucCoords.getRawPtr(), qVec);
		const xyz_type nearVec = eucCoords->getGlobalCoords(nearInd);
		const ST dist = eucCoords->globalDistance(nearInd, qVec);
		for (int i = 0; i < nProcs; ++i) {
			comm->barrier();
			if (procRank == i ) {	
				std::cout << "Proc " << procRank << " globalKnnSearch :" << std::endl;
				std::cout << "\tnearest particle to " << qVec << " is at " << nearVec << std::endl;
				std::cout << "\tdist = " << dist << std::endl;
				std::cout << "\tdist. sq. = " << dist * dist << std::endl;
			}
		}
		comm->barrier();
		std::cout << "GLOBAL DEMO COMPLETE " << procRank << std::endl;
	}
	}
	Kokkos::finalize();
return 0;
}
