#ifndef _COMPADRE_SPHERICAL_COORDST_HPP_
#define _COMPADRE_SPHERICAL_COORDST_HPP_

#include "Compadre_CoordsT.hpp"

namespace Compadre {

struct generate_random_sphere;

/**
 * Derived class of CoordsT specialized for spherical coordinates.
*/

class SphericalCoordsT : public CoordsT {

	protected :
	
	public :

		SphericalCoordsT(const global_index_type n, const Teuchos::RCP<const Teuchos::Comm<int> >& comm, const local_index_type nDim = 3) : 
			CoordsT(n, comm, nDim) {};
		virtual ~SphericalCoordsT() {};

		friend class ParticlesT;
		friend class LagrangianParticlesT;
	
		scalar_type distance(const local_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords = true) const;
		
		scalar_type distance(const CoordsT::xyz_type queryPt1, const CoordsT::xyz_type queryPt2) const;

		scalar_type globalDistance(const global_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords = true) const;
			
		CoordsT::xyz_type midpoint(const local_index_type idx1, const CoordsT::xyz_type queryPt, bool use_physical_coords = true) const;
		
		CoordsT::xyz_type centroid(const std::vector<CoordsT::xyz_type> vecs) const;
		
        scalar_type triArea(const CoordsT::xyz_type vecA, const CoordsT::xyz_type vecB, const CoordsT::xyz_type vecC) const;
		
		scalar_type latitude(const local_index_type idx, bool use_physical_coords = true) const;
		scalar_type colatitude(const local_index_type idx, bool use_physical_coords = true) const;
		scalar_type longitude(const local_index_type idx, bool use_physical_coords = true) const;
		
		virtual void initRandom(const scalar_type s, const local_index_type i, bool use_physical_coords = true);
};

}

#endif
