#include "Compadre_nanoflannInformation.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#ifdef COMPADRE_USE_NANOFLANN
#include "Compadre_nanoflannPointCloudT.hpp"
#endif

#ifdef COMPADRE_USE_OPENMP
#include <omp.h>
#endif

#ifdef COMPADRE_USE_NANOFLANN
namespace Compadre {

NanoFlannInformation::NanoFlannInformation( const particles_type * source_particles, Teuchos::RCP<Teuchos::ParameterList> parameters
		, local_index_type maxLeaf, const particles_type * target_particles, bool use_physical_coords) :
				_maxLeaf(maxLeaf), NeighborhoodT(source_particles, parameters, target_particles)
{
	if (source_particles->getCoordsConst()->getComm()->getSize()>1) {
		// user Zoltan2 processor boundaries for neighbor search
		_cloud = Teuchos::rcp(new cloud_type(source_particles->getCoordsConst(), use_physical_coords, true));
	} else {
		// let nanoflann determine single processor boundary
		_cloud = Teuchos::rcp(new cloud_type(source_particles->getCoordsConst(), use_physical_coords, false));
	}
	_index = Teuchos::rcp(new tree_type(_nDim, *(_cloud), nanoflann::KDTreeSingleIndexAdaptorParams(_maxLeaf)));
	_index->buildIndex();
}

void NanoFlannInformation::constructSingleNeighborList(const scalar_type* coordinate,
		const scalar_type radius, std::vector<std::pair<size_t, scalar_type> >& neighbors, bool use_physical_coords) const {

	nanoflann::SearchParams sp; // default parameters
	_index->radiusSearch(&coordinate[0], radius*radius, neighbors, sp) ;

//	if (_source_coords->getComm()->getRank()==0) std::cout << "radius: " << radius << std::endl;
	for (int i=0, n=neighbors.size(); i<n; i++) {
		neighbors[i].second = std::sqrt(neighbors[i].second);
//		const Compadre::XyzVector current_coord(coordinate);
//		const size_t other_id = neighbors[i].first;
//		const Compadre::XyzVector other_coord = _source_coords->getLocalCoords((local_index_type)other_id, true);
//		if (_source_coords->getComm()->getRank()==0) std::cout << "i: " << i << " my_pt: " << current_coord << " dist: " << _source_coords->distance((local_index_type)other_id, current_coord) << " ot_pt: " << other_coord << std::endl;
//		if (_source_coords->getComm()->getRank()==0) std::cout << other_id << ", vs " << neighbors[i].second << std::endl;
	}
}

}
#endif
