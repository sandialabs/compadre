#ifndef _COMPADRE_LAGRANGIANSHALLOWWATERPHYSICS_HPP_
#define _COMPADRE_LAGRANGIANSHALLOWWATERPHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

class ParticlesT;

class LagrangianShallowWaterPhysics : public PhysicsT {

	protected:

		typedef Compadre::ParticlesT particle_type;
		
		double _gravity;

	public:

		LagrangianShallowWaterPhysics(	Teuchos::RCP<particle_type> particles,
								Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
								Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
									PhysicsT(particles, A_graph, A) {
			_gravity = 0;
		}

		void setGravity(const scalar_type gravity) { _gravity = gravity; }

		virtual double getStableTimeStep(double mesh_size, double H) {
			return mesh_size / std::sqrt(_gravity * H);
		}

		virtual ~LagrangianShallowWaterPhysics() {}

		virtual void computeVector(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0, scalar_type current_timestep_size = 0.0, scalar_type previous_timestep_size = -1.0);

		virtual const std::vector<InteractingFields> gatherFieldInteractions();
    
};

}
#endif
