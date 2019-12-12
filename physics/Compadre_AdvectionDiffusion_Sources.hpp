#ifndef _COMPADRE_ADVECTIONDIFFUSIONSOURCES_HPP_
#define _COMPADRE_ADVECTIONDIFFUSIONSOURCES_HPP_

#include <Compadre_SourcesT.hpp>
#include <Compadre_XyzVector.hpp>

namespace Compadre {

class AdvectionDiffusionPhysics;

class AdvectionDiffusionSources : public SourcesT {

	protected:

		typedef Compadre::ParticlesT particle_type;

        AdvectionDiffusionPhysics* _physics;

	public:

		AdvectionDiffusionSources(	Teuchos::RCP<particle_type> particles,
										mvec_type* b = NULL) :
										SourcesT(particles, b) {};

		virtual ~AdvectionDiffusionSources() {};

		virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();

        void setPhysics(Teuchos::RCP<AdvectionDiffusionPhysics> physics) { _physics = physics.getRawPtr(); }

};

}

#endif
