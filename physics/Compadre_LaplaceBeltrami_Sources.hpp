#ifndef _COMPADRE_LAPLACE_BELTRAMI_SOURCES_HPP_
#define _COMPADRE_LAPLACE_BELTRAMI_SOURCES_HPP_

#include <Compadre_SourcesT.hpp>

namespace Compadre {

class LaplaceBeltramiSources : public SourcesT {

	protected:

		typedef Compadre::ParticlesT particle_type;

	public:

		LaplaceBeltramiSources(	Teuchos::RCP<particle_type> particles,
										mvec_type* b = NULL) :
										SourcesT(particles, b)
		{}

		virtual ~LaplaceBeltramiSources() {};

		virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0, scalar_type current_timestep_size = 0.0, scalar_type previous_timestep_size = -1.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();

};

}

#endif
