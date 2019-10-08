#ifndef _COMPADRE_ADVECTIONDIFFUSIONPHYSICS_HPP_
#define _COMPADRE_ADVECTIONDIFFUSIONPHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>
#include <Compadre_XyzVector.hpp>

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

class ParticlesT;

class AdvectionDiffusionPhysics : public PhysicsT {

	protected:

		typedef Compadre::ParticlesT particle_type;
        typedef Compadre::XyzVector xyz_type;
		local_index_type Porder;

        xyz_type _advection_field;
        scalar_type _diffusion;
        
        particle_type* _cells;
		
	public:

		AdvectionDiffusionPhysics(	Teuchos::RCP<particle_type> particles, local_index_type t_Porder,
								Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
								Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
									PhysicsT(particles, A_graph, A), Porder(t_Porder) {}
		
		virtual ~AdvectionDiffusionPhysics() {}
		
		virtual Teuchos::RCP<crs_graph_type> computeGraph(local_index_type field_one, local_index_type field_two = -1);

		virtual void computeMatrix(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual const std::vector<InteractingFields> gatherFieldInteractions();

        void setAdvectionField(const xyz_type advection_field) {
            for (local_index_type i=0; i<3; ++i) {
                _advection_field[i] = advection_field[i];
            }
        }
        
        xyz_type getAdvectionField() const { return _advection_field; }

        void setDiffusion(const scalar_type diffusion) {
            _diffusion = diffusion;
        }

        scalar_type getDiffusion() const { return _diffusion; }

        void setCells(Teuchos::RCP<particle_type> cells) { _cells = cells.getRawPtr(); };
    
};

}
#endif
