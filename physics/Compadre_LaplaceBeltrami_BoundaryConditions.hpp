#ifndef _COMPADRE_LAPLACE_BELTRAMI_BOUNDARYCONDITIONS_HPP_
#define _COMPADRE_LAPLACE_BELTRAMI_BOUNDARYCONDITIONS_HPP_

#include <Compadre_BoundaryConditionsT.hpp>

namespace Compadre {

class ParticlesT;

class LaplaceBeltramiBoundaryConditions : public BoundaryConditionsT {

	protected:

		typedef Compadre::ParticlesT particle_type;

		local_index_type _physics_type;

	public:

		LaplaceBeltramiBoundaryConditions( Teuchos::RCP<particle_type> particles,
												mvec_type* b = NULL) :
												BoundaryConditionsT(particles, b)
		{}

		virtual ~LaplaceBeltramiBoundaryConditions() {}

		virtual void flagBoundaries();

		virtual void applyBoundaries(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0, scalar_type current_timestep_size = 0.0, scalar_type previous_timestep_size = -1.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();

		void setPhysicsType(const local_index_type physics_type) { _physics_type = physics_type; }
};

}

#endif
