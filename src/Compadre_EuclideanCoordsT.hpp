#ifndef _COMPADRE_EUCLIDEAN_COORDST_HPP_
#define _COMPADRE_EUCLIDEAN_COORDST_HPP_

#include "Compadre_CoordsT.hpp"

namespace Compadre {

struct generate_random_3d;

/**
	Derived class of CoordsT specialized for Euclidean coordinates.
*/

class EuclideanCoordsT : public CoordsT {

	protected :
	
	public :

		EuclideanCoordsT(const global_index_type n, const Teuchos::RCP<const Teuchos::Comm<int> >& comm, const local_index_type nDim = 3) : 
			CoordsT(n, comm, nDim) {};
		virtual ~EuclideanCoordsT() {};
	
		friend class ParticlesT;
		friend class LagrangianParticlesT;

		virtual scalar_type distance(const local_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords = true) const;
		
		virtual scalar_type distance(const CoordsT::xyz_type queryPt1, const CoordsT::xyz_type queryPt2) const;

		virtual scalar_type globalDistance(const global_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords = true) const;
			
		virtual CoordsT::xyz_type midpoint(const local_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords = true) const;
		
		virtual CoordsT::xyz_type centroid(const std::vector<CoordsT::xyz_type> vecs) const;
		
		virtual scalar_type triArea(const CoordsT::xyz_type vecA, const CoordsT::xyz_type vecB, const CoordsT::xyz_type vecC) const;
		
		virtual void initRandom(const scalar_type s, const local_index_type i, bool use_physical_coords = true);
};

}

#endif
