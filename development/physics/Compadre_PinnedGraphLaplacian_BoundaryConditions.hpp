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
												Teuchos::RCP<mvec_type> b = Teuchos::null) :
												BoundaryConditionsT(particles, b)
		{}

		virtual ~PinnedGraphLaplacianBoundaryConditions() {}

		virtual void flagBoundaries();

		virtual void applyBoundaries(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();
};

}

#endif

