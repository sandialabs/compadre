#include "Compadre_EuclideanCoordsT.hpp"

#include "Compadre_XyzVector.hpp"

namespace Compadre {

struct generate_random_3d {

	device_view_type vals;
	pool_type pool;
	scalar_type maxRange;

	generate_random_3d(device_view_type vals_, const scalar_type range, const int seedPlus ) :
		vals(vals_), pool(5374857+seedPlus), maxRange(range) {};

	KOKKOS_INLINE_FUNCTION
	void operator() (int i) const {
		generator_type rand_gen = pool.get_state();
		for (int j = 0; j < 3; ++j ){
			vals(i,j) = rand_gen.drand(maxRange);
		}
		pool.free_state(rand_gen);
	}
};

scalar_type EuclideanCoordsT::distance(const local_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords) const {
	xyz_type vecA = this->getLocalCoords(idx1, true /*use halo*/, use_physical_coords);
	return euclideanDistance(vecA, queryPt);
}

scalar_type EuclideanCoordsT::distance( const CoordsT::xyz_type queryPt1, const CoordsT::xyz_type queryPt2) const {
	return euclideanDistance(queryPt1, queryPt2);
}

scalar_type EuclideanCoordsT::globalDistance(const global_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords) const {
	xyz_type vecA = this->getGlobalCoords(idx1, use_physical_coords);
	return euclideanDistance(vecA, queryPt);
}

CoordsT::xyz_type EuclideanCoordsT::midpoint(const local_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords) const {
	xyz_type vecA = this->getLocalCoords(idx1, true /*use halo*/, use_physical_coords);
	return euclideanMidpoint(vecA, queryPt);
}

CoordsT::xyz_type EuclideanCoordsT::centroid(const std::vector<CoordsT::xyz_type> vecs) const {
	return euclideanCentroid(vecs);
}

scalar_type EuclideanCoordsT::triArea(const CoordsT::xyz_type vecA, const CoordsT::xyz_type vecB, const CoordsT::xyz_type vecC) const {
	return euclideanTriArea(vecA, vecB, vecC);
}

void EuclideanCoordsT::initRandom(const scalar_type s, const local_index_type i, bool use_physical_coords) {
	mvec_type* pts_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	device_view_type ptsView = pts_vec->getLocalView<device_view_type>();
	Kokkos::parallel_for(this->nLocalMax(),
		generate_random_3d(ptsView, s, i) );
		this->setLocalN(this->nLocalMax());
		pts_vec->modify<device_view_type>();
		pts_vec->sync<host_memory_space>();
}

}

