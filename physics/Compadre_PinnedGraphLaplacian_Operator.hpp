#ifndef _COMPADRE_PINNEDGRAPHLAPLACIANPHYSICS_HPP_
#define _COMPADRE_PINNEDGRAPHLAPLACIANPHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>

namespace Compadre {

class PinnedGraphLaplacianPhysics : public PhysicsT {

	protected:

	public:

		PinnedGraphLaplacianPhysics(	Teuchos::RCP<particle_type> particles,
										Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
										Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
										PhysicsT(particles, A_graph, A)
		{}

		virtual ~PinnedGraphLaplacianPhysics() {}

		virtual Teuchos::RCP<crs_graph_type> computeGraph(local_index_type field_one, local_index_type field_two = -1);

		virtual void computeMatrix(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0, scalar_type current_timestep_size = 0.0, scalar_type previous_timestep_size = -1.0);

		virtual const std::vector<InteractingFields> gatherFieldInteractions();

};

}

#endif
