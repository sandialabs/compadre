#ifndef _COMPADRE_LAPLACE_BELTRAMI_PHYSICS_HPP_
#define _COMPADRE_LAPLACE_BELTRAMI_PHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

class ParticlesT;

class LaplaceBeltramiPhysics : public PhysicsT {

	protected:

		typedef Compadre::ParticlesT particle_type;
		local_index_type Porder;
		local_index_type _physics_type;
		
	public:

		LaplaceBeltramiPhysics(	Teuchos::RCP<particle_type> particles, local_index_type t_Porder,
								Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
								Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
									PhysicsT(particles, A_graph, A), Porder(t_Porder), _physics_type(0) {}
		
		virtual ~LaplaceBeltramiPhysics() {}
		
		virtual Teuchos::RCP<crs_graph_type> computeGraph(local_index_type field_one, local_index_type field_two = -1);

		virtual void computeMatrix(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual const std::vector<InteractingFields> gatherFieldInteractions();

		void setPhysicsType(const local_index_type physics_type) { _physics_type = physics_type; }
    
};

}
#endif
