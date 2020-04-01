#ifndef _COMPADRE_REACTIONDIFFUSIONBOUNDARYCONDITIONS_HPP_
#define _COMPADRE_REACTIONDIFFUSIONBOUNDARYCONDITIONS_HPP_

#include <Compadre_BoundaryConditionsT.hpp>

namespace Compadre {

class ParticlesT;

class ReactionDiffusionPhysics;

class ReactionDiffusionBoundaryConditions : public BoundaryConditionsT {

    protected:

        typedef Compadre::ParticlesT particle_type;

        ReactionDiffusionPhysics* _physics;

    public:

        ReactionDiffusionBoundaryConditions( Teuchos::RCP<particle_type> particles,
                                                mvec_type* b = NULL) :
                                                BoundaryConditionsT(particles, b)
        {}

        virtual ~ReactionDiffusionBoundaryConditions() {}

        virtual void flagBoundaries();

        virtual void applyBoundaries(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

        virtual std::vector<InteractingFields> gatherFieldInteractions();

        void setPhysics(Teuchos::RCP<ReactionDiffusionPhysics> physics) { _physics = physics.getRawPtr(); }

};

}

#endif
