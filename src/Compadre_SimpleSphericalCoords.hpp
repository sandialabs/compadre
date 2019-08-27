#ifndef _COMPADRE_SIMPLE_SPHERICAL_COORDS_HPP_
#define _COMPADRE_SIMPLE_SPHERICAL_COORDS_HPP_

#include "Compadre_SimpleCoords.hpp"

namespace Compadre { 

struct generate_random_sphere {
    typedef Kokkos::View<scalar_type*[3]> view_type;
    typedef Kokkos::Random_XorShift64_Pool<> pool_type;
    typedef pool_type::generator_type generator_type;
    
    view_type vals;
    scalar_type sphRadius;
    pool_type pool;
    
    generate_random_sphere(view_type vals_, const scalar_type sphR = 1.0, const local_index_type seedPlus = 0) :
        vals(vals_), sphRadius(sphR), pool(1234+seedPlus) {};
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const local_index_type i) const {
        generator_type rand_gen = pool.get_state();
        scalar_type u = rand_gen.drand(-1.0, 1.0);
        scalar_type v = rand_gen.drand(-1.0, 1.0);
        while (u*u + v*v > 1.0) {
            u = rand_gen.drand(-1.0,1.0);
            v = rand_gen.drand(-1.0,1.0);
        }
        vals(i,0) = sphRadius * (2.0 * u * std::sqrt(1.0 - u*u - v*v));
        vals(i,1) = sphRadius * (2.0 * v * std::sqrt(1.0 - u*u - v*v));
        vals(i,2) = sphRadius * (1.0 - 2.0 * (u*u + v*v));
        pool.free_state(rand_gen);
    }
};

class SimpleSphericalCoords : public SimpleCoords {
    protected:
        typedef Kokkos::View<scalar_type*[3]> view_type;
        typedef XyzVector xyz_type;
        
        scalar_type sphRadius;
    public:
        SimpleSphericalCoords(const local_index_type nMax, const std::string id="", const scalar_type sphR = 1.0) :
            SimpleCoords(nMax, id), sphRadius(sphR) {};
        
        virtual ~SimpleSphericalCoords() {};
        
        scalar_type distance(const local_index_type idx, const xyz_type& otherPt) const {
            return sphereDistance(this->xyz(idx), otherPt, sphRadius);
        }
        scalar_type distance(const local_index_type idx1, const local_index_type idx2) const {
            return sphereDistance(this->xyz(idx1), this->xyz(idx2), sphRadius);
        }
        xyz_type midpoint(const local_index_type idx, const xyz_type& otherPt) const {
            return sphereMidpoint(this->xyz(idx), otherPt, sphRadius);
        }
        xyz_type midpoint(const local_index_type idx1, const local_index_type idx2) const {
            return sphereMidpoint(this->xyz(idx1), this->xyz(idx2), sphRadius);
        }
        xyz_type centroid(const std::vector<xyz_type>& vecs) const {
            return sphereCentroid(vecs, sphRadius);
        }
        xyz_type centroid(const std::vector<local_index_type>& inds) const {
            std::vector<xyz_type> vecs;
            for (size_t i = 0; i < inds.size(); ++i)
                vecs.push_back(this->xyz(inds[i]));
            return sphereCentroid(vecs, sphRadius);
        }
        scalar_type triArea(const xyz_type& vecA, const xyz_type& vecB, const xyz_type& vecC) const {
            return sphereTriArea(vecA, vecB, vecC, sphRadius);
        }
        scalar_type triArea(const std::vector<xyz_type>& vecs) const {
            return sphereTriArea(vecs[0], vecs[1], vecs[2], sphRadius);
        }
        scalar_type triArea(const local_index_type idx1, const local_index_type idx2, const local_index_type idx3) const {
            return sphereTriArea(this->xyz(idx1), this->xyz(idx2), this->xyz(idx3), sphRadius);
        }
        
        void initRandom(const scalar_type scr, const local_index_type idx) {
            Kokkos::parallel_for(this->_nMax, 
                generate_random_sphere(this->points, scr, idx));
            this->_nUsed = this->_nMax;
        }
};

}
#endif
