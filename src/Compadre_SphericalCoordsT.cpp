#include "Compadre_SphericalCoordsT.hpp"

#include "Compadre_XyzVector.hpp"

namespace Compadre {

struct generate_random_sphere {

	device_view_type vals;
	pool_type pool;
	scalar_type sphRadius;

	generate_random_sphere(device_view_type vals_, const scalar_type sphRadius_ = 1.0, const int seedPlus = 0) :
		vals(vals_), pool(1234+seedPlus), sphRadius(sphRadius_) {};

	KOKKOS_INLINE_FUNCTION
	void operator () (const int i) const {
		generator_type rand_gen = pool.get_state();
		scalar_type u = rand_gen.drand(-1.0, 1.0);
		scalar_type v = rand_gen.drand(-1.0, 1.0);
		while (u*u + v*v > 1.0 ) {
			u = rand_gen.drand(-1.0, 1.0);
			v = rand_gen.drand(-1.0, 1.0);
		}
		vals(i,0) = sphRadius * (2.0 * u * std::sqrt(1.0 - u*u - v*v));
		vals(i,1) = sphRadius * (2.0 * v * std::sqrt(1.0 - u*u - v*v));
		vals(i,2) = sphRadius * (1.0 - 2.0 * (u*u + v*v));
		pool.free_state(rand_gen);
	}
};

scalar_type SphericalCoordsT::distance(const local_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords) const {
	xyz_type vecA = this->getLocalCoords(idx1, true /*use halo*/, use_physical_coords);
	return sphereDistance(vecA, queryPt);
}

scalar_type SphericalCoordsT::distance(const CoordsT::xyz_type queryPt1, const CoordsT::xyz_type queryPt2) const {
	return sphereDistance(queryPt1, queryPt2);
}

scalar_type SphericalCoordsT::globalDistance(const global_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords) const {
	xyz_type vecA = this->getGlobalCoords(idx1, use_physical_coords);
	return sphereDistance(vecA, queryPt);
}

CoordsT::xyz_type SphericalCoordsT::midpoint(const local_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords) const {
	xyz_type vecA = this->getLocalCoords(idx1, true /*use halo*/, use_physical_coords);
	return sphereMidpoint(vecA, queryPt);
}

CoordsT::xyz_type SphericalCoordsT::centroid(const std::vector<CoordsT::xyz_type> vecs) const {
	return sphereCentroid(vecs);
}

scalar_type SphericalCoordsT::triArea(const CoordsT::xyz_type vecA, const CoordsT::xyz_type vecB, const CoordsT::xyz_type vecC) const {
	return sphereTriArea(vecA, vecB, vecC);
}

scalar_type SphericalCoordsT::latitude(const local_index_type idx, bool use_physical_coords) const {
    xyz_type vec = this->getLocalCoords(idx, true /*use halo*/, use_physical_coords);
    return vec.latitude();
}

scalar_type SphericalCoordsT::colatitude(const local_index_type idx, bool use_physical_coords) const {
    xyz_type vec = this->getLocalCoords(idx, true /*use halo*/, use_physical_coords);
    return vec.colatitude();
}

scalar_type SphericalCoordsT::longitude(const local_index_type idx, bool use_physical_coords) const {
    xyz_type vec = this->getLocalCoords(idx, true /*use halo*/, use_physical_coords);
    return vec.longitude();
}

void SphericalCoordsT::initRandom(const scalar_type s, const local_index_type i, bool use_physical_coords) {
	mvec_type* pts_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	device_view_type ptsView = pts_vec->getLocalView<device_view_type>();
	Kokkos::parallel_for(this->nLocalMax(),
		generate_random_sphere(ptsView, s, i) );
		this->setLocalN(this->nLocalMax());
		pts_vec->modify<device_view_type>();
		pts_vec->sync<host_memory_space>();
}

}

