#ifndef _COMPADRE_PINNEDGRAPHLAPLACIANBOUNDARYCONDITIONS_HPP_
#define _COMPADRE_PINNEDGRAPHLAPLACIANBOUNDARYCONDITIONS_HPP_

#include <Compadre_BoundaryConditionsT.hpp>

namespace Compadre {

class ParticlesT;

class PinnedGraphLaplacianBoundaryConditions : public BoundaryConditionsT {

	protected:

		typedef Compadre::ParticlesT particle_type;

	public:

		PinnedGraphLaplacianBoundaryConditions( Teuchos::RCP<particle_type> particles,
												mvec_type* b = NULL) :
												BoundaryConditionsT(particles, b)
		{}

		virtual ~PinnedGraphLaplacianBoundaryConditions() {}

		virtual void flagBoundaries();

		virtual void applyBoundaries(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0, scalar_type current_timestep_size = 0.0, scalar_type previous_timestep_size = -1.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();
};

}

#endif

