#ifndef _COMPADRE_GMLS_POISSONNEUMANNBOUNDARYCONDITIONS_
#define _COMPADRE_GMLS_POISSONNEUMANNBOUNDARYCONDITIONS_

#include <Compadre_BoundaryConditionsT.hpp>

namespace Compadre {

class ParticlesT;

class GMLS_PoissonNeumannBoundaryConditions : public BoundaryConditionsT {
    protected:
        typedef Compadre::ParticlesT particle_type;

    public:
        GMLS_PoissonNeumannBoundaryConditions(Teuchos::RCP<particle_type> particles,
            mvec_type* b = NULL) : BoundaryConditionsT(particles, b) {}

        virtual ~GMLS_PoissonNeumannBoundaryConditions() {}

        virtual void flagBoundaries();

        virtual void applyBoundaries(local_index_type field_one, local_index_type field_two=-1, scalar_type time=0.0, scalar_type current_timestep_size = 0.0, scalar_type previous_timestep_size = -1.0);

        virtual std::vector<InteractingFields> gatherFieldInteractions();
};

}

#endif
