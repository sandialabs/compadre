#ifndef _COMPADRE_ADVECTIONDIFFUSIONSOURCES_HPP_
#define _COMPADRE_ADVECTIONDIFFUSIONSOURCES_HPP_

#include <Compadre_SourcesT.hpp>

namespace Compadre {

class AdvectionDiffusionSources : public SourcesT {

	protected:

		typedef Compadre::ParticlesT particle_type;

	public:

		AdvectionDiffusionSources(	Teuchos::RCP<particle_type> particles,
										Teuchos::RCP<mvec_type> b = Teuchos::null) :
										SourcesT(particles, b)
		{}

		virtual ~AdvectionDiffusionSources() {};

		virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();

};

}

#endif
