#ifndef _COMPADRE_GMLS_PINNEDLAPLACIANSOURCES_HPP_
#define _COMPADRE_GMLS_PINNEDLAPLACIANSOURCES_HPP_

#include <Compadre_SourcesT.hpp>

namespace Compadre {

class GMLS_PinnedLaplacianSources : public SourcesT {

	protected:

		typedef Compadre::ParticlesT particle_type;

	public:

		GMLS_PinnedLaplacianSources(	Teuchos::RCP<particle_type> particles,
										mvec_type* b = NULL) :
										SourcesT(particles, b)
		{}

		virtual ~GMLS_PinnedLaplacianSources() {};

		virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();

};

}

#endif
