#ifndef _COMPADRE_SIMPLE_COORDS_HPP_
#define _COMPADRE_SIMPLE_COORDS_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"
#include "Compadre_XyzVector.hpp"

namespace Compadre {

/* Simple Coords are meant to be used when the entire system can fit on one node (i.e., storage is not distributed).
   Distributed work is achieved with the "replicated data" approach -- each node maintains its own copy of entire system.
*/
class SimpleCoords {
    protected:
        typedef Kokkos::View<scalar_type*[3]> view_type;
        typedef XyzVector xyz_type;
    
        local_index_type _nMax;
        local_index_type _nUsed;
        
    public:
        view_type points;
        
        SimpleCoords(const local_index_type nMax, const std::string id) : points(id, nMax), _nMax(nMax), _nUsed(0) {};
        
        virtual ~SimpleCoords() {};
        
        local_index_type nMax() const {return _nMax;}
        local_index_type nUsed() const {return _nUsed;}
        inline xyz_type xyz(const local_index_type idx) const {return xyz_type(points(idx,0), points(idx,1), points(idx,2));}
        
        virtual scalar_type distance(const local_index_type idx, const xyz_type& otherPt) const = 0;
        virtual scalar_type distance(const local_index_type idx1, const local_index_type idx2) const = 0;
        virtual xyz_type midpoint(const local_index_type idx1, const local_index_type idx2) const = 0;
        virtual xyz_type midpoint(const local_index_type idx, const xyz_type& otherPt) const = 0;
        virtual xyz_type centroid(const std::vector<xyz_type>& vecs) const = 0;
        virtual xyz_type centroid(const std::vector<local_index_type>& inds) const = 0;
        virtual scalar_type triArea(const xyz_type& vecA, const xyz_type& vecB, const xyz_type& vecC) const = 0;
        virtual scalar_type triArea(const std::vector<xyz_type>& vecs) const = 0;
        virtual scalar_type triArea(const local_index_type idx1, const local_index_type idx2, const local_index_type idx3) const = 0;
        
        virtual void initRandom(const scalar_type scr, const local_index_type idx) = 0;
        
        xyz_type crossProduct(const local_index_type idx, const xyz_type& otherPt) const {
            return xyz_type( points(idx,1) * otherPt.z - points(idx,2) * otherPt.y,
                             points(idx,2) * otherPt.x - points(idx,0) * otherPt.z,
                             points(idx,0) * otherPt.y - points(idx,1) * otherPt.x);
        }
        
        xyz_type crossProduct(const local_index_type idx1, const local_index_type idx2) const {
            return crossProduct(idx1, this->xyz(idx2));
        }
        
        scalar_type dotProduct(const local_index_type idx, const xyz_type& otherPt) const {
            return points(idx,0) * otherPt.x + points(idx,1) * otherPt.y + points(idx,2) * otherPt.z;
        }
        
        scalar_type dotProduct(const local_index_type idx1, const local_index_type idx2) const {
            return dotProduct(idx1, xyz(idx2));
        }
        
        void setXyz(const local_index_type idx, const xyz_type& newVec) {
            TEUCHOS_TEST_FOR_EXCEPTION( (idx <0) || (idx >= _nMax), std::out_of_range, 
                "Error: index " << idx << " is out of bounds for nMax = " << _nMax  << ".");
            points(idx,0) = newVec.x;
            points(idx,1) = newVec.y;
            points(idx,2) = newVec.z;
        }
        
        void insertXyz(const xyz_type& newVec) {
            TEUCHOS_TEST_FOR_EXCEPTION( _nUsed + 1 > _nMax, std::out_of_range,
                "Error: cannot insert more coordinates; nUsed = " << _nUsed << ", nMax = " << _nMax);
            points(_nUsed, 0) = newVec.x;
            points(_nUsed, 1) = newVec.y;
            points(_nUsed++, 2) = newVec.z;
        }
        
        void insertXyz(const scalar_type x = 0.0, const scalar_type y = 0.0, const scalar_type z = 0.0) {
            TEUCHOS_TEST_FOR_EXCEPTION(_nUsed + 1 > _nMax, std::out_of_range,
                "Error: cannot insert more coordinates; nUsed = " << _nUsed << ", nMax = " << _nMax);
            points(_nUsed,0) = x;
            points(_nUsed,1) = y;
            points(_nUsed++,2) = z;
        }
        
        void writeToMatlab(std::ostream& fs, const std::string coordsName, const int procRank) {
            fs << coordsName << "Xyz" << procRank << " = [";
            for (local_index_type i = 0; i < _nUsed - 1; ++i)
                fs << points(i,0) << ", " << points(i,1) << ", " << points(i,2) << ";..." << std::endl;
            fs << points(_nUsed-1,0) << ", " << points(_nUsed-1,1) << ", " << points(_nUsed-1,2) << "];" << std::endl;
        }
        
        scalar_type magnitude(const local_index_type idx) const {
            return xyz(idx).magnitude();
        }
};

std::ostream& operator << (std::ostream& os, 
    const Compadre::SimpleCoords& coords) {
    os << "SimpleCoords basic info:\n";
    os << "\tnMax = " << coords.nMax() << std::endl;
    os << "\tnUsed = " << coords.nUsed() << std::endl;
    return os;
}

}
#endif
