#ifndef _COMPADRE_NANOFLANNINFORMATION_HPP_
#define _COMPADRE_NANOFLANNINFORMATION_HPP_

#include "Compadre_NeighborhoodT.hpp"
#include "TPL/nanoflann/nanoflann.hpp"

namespace Compadre {

class CoordsT;
class ParticlesT;
class LocalPointCloudT;

/**
 * Derived class of NeighborhoodT specialized for using Nanoflann to perform KdTree search.
*/

class NanoFlannInformation : public NeighborhoodT {

	protected:

		typedef Compadre::LocalPointCloudT cloud_type;
		typedef nanoflann::KDTreeSingleIndexAdaptor<
					nanoflann::L2_Simple_Adaptor<scalar_type, cloud_type>, cloud_type, 3> tree_type;

		Teuchos::RCP<cloud_type> _cloud;
		Teuchos::RCP<tree_type> _index;

		local_index_type _maxLeaf;

	public:

		NanoFlannInformation(const particles_type * source_particles,
							Teuchos::RCP<Teuchos::ParameterList> parameters,
							local_index_type maxLeaf = 10, const particles_type * target_particles = NULL, bool use_physical_coords = true);

		virtual ~NanoFlannInformation() {};

		virtual void constructSingleNeighborList(const scalar_type* coordinate,
				const scalar_type radius, std::vector<std::pair<size_t, scalar_type> >& neighbors, bool use_physical_coords = true) const;


};


}
#endif
