#ifndef _COMPADRE_GMLS_STAGGEREDINTEGRAL_PINNEDLAPLACIANPHYSICS_HPP_
#define _COMPADRE_GMLS_STAGGEREDINTEGRAL_PINNEDLAPLACIANPHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>

/*
 Constructs the matrix operator.

 */
namespace Compadre {

class ParticlesT;

class GMLS_StaggeredIntegral_LaplacianPhysics : public PhysicsT {

	protected:
		typedef Compadre::ParticlesT particle_type;
		local_index_type Porder;

	public:
		GMLS_StaggeredIntegral_LaplacianPhysics(Teuchos::RCP<particle_type> particles, local_index_type t_Porder,
                                                Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
                                                Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
                PhysicsT(particles, A_graph, A), Porder(t_Porder) {}

		virtual ~GMLS_StaggeredIntegral_LaplacianPhysics() {}
		virtual Teuchos::RCP<crs_graph_type> computeGraph(local_index_type field_one, local_index_type field_two = -1);
		virtual void computeMatrix(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);
		virtual const std::vector<InteractingFields> gatherFieldInteractions();
};

}
#endif
