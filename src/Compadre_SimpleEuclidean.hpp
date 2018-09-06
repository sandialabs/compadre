#ifndef _COMPADRE_SIMPLE_EUCLIDEAN_COORDS_HPP_
#define _COMPADRE_SIMPLE_EUCLIDEAN_COORDS_HPP_

#include "Compadre_SimpleCoords.hpp"

namespace Compadre {

struct generate_random_3d {
    typedef Kokkos::View<scalar_type*[3]> view_type;
    typedef Kokkos::Random_XorShift64_Pool<> pool_type;
    typedef pool_type::generator_type generator_type;
    
    view_type vals;
    scalar_type maxRange;
    pool_type pool;
    
    generate_random_3d(view_type vals_, const scalar_type range, const int seedPlus) :
        vals(vals_), pool(5374857+seedPlus), maxRange(range) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (local_index_type i) const {
        generator_type rand_gen = pool.get_state();
        for (int j = 0; j < 3; ++j) 
            vals(i,j) = rand_gen.drand(maxRange);
        pool.free_state(rand_gen);
    }
};

class SimpleEuclideanCoords : public SimpleCoords
{
    protected:
        typedef Kokkos::View<scalar_type*[3]> view_type;
        typedef XyzVector xyz_type;
    
    public:
        SimpleEuclideanCoords(const local_index_type nMax, const std::string id = "coords") : 
            SimpleCoords(nMax, id) {};
        virtual ~SimpleEuclideanCoords() {};
        
        scalar_type distance(const local_index_type idx, const xyz_type& otherPt) const {
            return euclideanDistance(this->xyz(idx), otherPt);
        }
        scalar_type distance(const local_index_type idx1, const local_index_type idx2) const {
            return euclideanDistance(this->xyz(idx1), this->xyz(idx2));
        }
        xyz_type midpoint(const local_index_type idx, const xyz_type& otherPt) const {
            return euclideanMidpoint(this->xyz(idx), otherPt);
        }
        xyz_type midpoint(const local_index_type idx1, const local_index_type idx2) const {
            return euclideanMidpoint(this->xyz(idx1), this->xyz(idx2));
        }
        xyz_type centroid(const std::vector<xyz_type>& vecs) const {
            return euclideanCentroid(vecs);
        }
        xyz_type centroid(const std::vector<local_index_type>& inds) const {
            std::vector<xyz_type> vecs;
            for (local_index_type i = 0; i < inds.size(); ++i)
                vecs.push_back(this->xyz(inds[i]));
            return euclideanCentroid(vecs);
        }
        scalar_type triArea(const xyz_type& vecA, const xyz_type& vecB, const xyz_type& vecC) const {
            return euclideanTriArea(vecA, vecB, vecC);
        }
        scalar_type triArea(const std::vector<xyz_type>& vecs) const {
            return euclideanTriArea(vecs[0], vecs[1], vecs[2]);
        }
        scalar_type triArea(const local_index_type idx1, const local_index_type idx2, const local_index_type idx3) const {
            return euclideanTriArea(this->xyz(idx1), this->xyz(idx2), this->xyz(idx3));
        }
        
        void initRandom(const scalar_type scr, const local_index_type idx) {
            Kokkos::parallel_for(this->_nMax,
                generate_random_3d(this->points, scr, idx));
            this->_nUsed = this->_nMax;
        }
    
};

}
#endif
