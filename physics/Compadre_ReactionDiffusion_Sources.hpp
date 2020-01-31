#ifndef _COMPADRE_REACTIONDIFFUSIONSOURCES_HPP_
#define _COMPADRE_REACTIONDIFFUSIONSOURCES_HPP_

#include <Compadre_SourcesT.hpp>
#include <Compadre_XyzVector.hpp>

namespace Compadre {

class ReactionDiffusionPhysics;

class ReactionDiffusionSources : public SourcesT {

	protected:

		typedef Compadre::ParticlesT particle_type;

        ReactionDiffusionPhysics* _physics;

	public:

		ReactionDiffusionSources(	Teuchos::RCP<particle_type> particles,
										mvec_type* b = NULL) :
										SourcesT(particles, b) {};

		virtual ~ReactionDiffusionSources() {};

		virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();

        void setPhysics(Teuchos::RCP<ReactionDiffusionPhysics> physics) { _physics = physics.getRawPtr(); }

};

}

#endif
