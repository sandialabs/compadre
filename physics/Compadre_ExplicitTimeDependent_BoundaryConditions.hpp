#ifndef _COMPADRE_EXPLICITTIMEDEPENDENTBOUNDARYCONDITIONS_HPP_
#define _COMPADRE_EXPLICITTIMEDEPENDENTBOUNDARYCONDITIONS_HPP_

#include <Compadre_BoundaryConditionsT.hpp>

namespace Compadre {

class ParticlesT;

class ExplicitTimeDependentPhysics;

class ExplicitTimeDependentBoundaryConditions : public BoundaryConditionsT {

	protected:

		typedef Compadre::ParticlesT particle_type;

        ExplicitTimeDependentPhysics* _physics;

        std::vector<int> _velocity_id, _acceleration_id;

	public:

		ExplicitTimeDependentBoundaryConditions( Teuchos::RCP<particle_type> particles,
												mvec_type* b = NULL) :
												BoundaryConditionsT(particles, b)
		{}

		virtual ~ExplicitTimeDependentBoundaryConditions() {}

		virtual void flagBoundaries();

		virtual void applyBoundaries(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0, scalar_type current_timestep_size = 0.0, scalar_type previous_timestep_size = -1.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();

        void setPhysics(Teuchos::RCP<ExplicitTimeDependentPhysics> physics) { _physics = physics.getRawPtr(); }

};

}

#endif
