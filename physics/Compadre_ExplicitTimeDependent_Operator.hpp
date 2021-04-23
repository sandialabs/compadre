#ifndef _COMPADRE_EXPLICITTIMEDEPENDENTPHYSICS_HPP_
#define _COMPADRE_EXPLICITTIMEDEPENDENTPHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

class ParticlesT;
class AnalyticFunction;

class ExplicitTimeDependentPhysics : public PhysicsT {

	protected:

		typedef Compadre::ParticlesT particle_type;
		
		double _gravity;

	public:

        Teuchos::RCP<AnalyticFunction> vel_source_function;
        Teuchos::RCP<AnalyticFunction> vel_boundary_function;
        Teuchos::RCP<AnalyticFunction> h_source_function;
        Teuchos::RCP<AnalyticFunction> h_boundary_function;

		ExplicitTimeDependentPhysics(	Teuchos::RCP<particle_type> particles,
								Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
								Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
									PhysicsT(particles, A_graph, A) {
			_gravity = 0;
		}

		void setGravity(const scalar_type gravity) { _gravity = gravity; }

		virtual double getStableTimeStep(double mesh_size, double H) {
			return mesh_size / std::sqrt(_gravity * H);
		}

		virtual ~ExplicitTimeDependentPhysics() {}

		virtual void computeVector(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0, scalar_type current_timestep_size = 0.0, scalar_type previous_timestep_size = -1.0);

		virtual const std::vector<InteractingFields> gatherFieldInteractions();
    
};

}
#endif
