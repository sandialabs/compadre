#ifndef _COMPADRE_ADVECTIONDIFFUSIONPHYSICS_HPP_
#define _COMPADRE_ADVECTIONDIFFUSIONPHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

class ParticlesT;

class AdvectionDiffusionPhysics : public PhysicsT {

	protected:

		typedef Compadre::ParticlesT particle_type;
		local_index_type Porder;
		
	public:

		AdvectionDiffusionPhysics(	Teuchos::RCP<particle_type> particles, local_index_type t_Porder,
								Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
								Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
									PhysicsT(particles, A_graph, A), Porder(t_Porder) {}
		
		virtual ~AdvectionDiffusionPhysics() {}
		
		virtual void computeGraph(local_index_type field_one, local_index_type field_two = -1);

		virtual void computeMatrix(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual const std::vector<InteractingFields> gatherFieldInteractions();
    
};

}
#endif
