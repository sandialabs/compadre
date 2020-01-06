#ifndef _COMPADRE_GMLS_STOKESBOUNDARYCONDITIONS_
#define _COMPADRE_GMLS_STOKESBOUNDARYCONDITIONS_

#include <Compadre_BoundaryConditionsT.hpp>

namespace Compadre {

class ParticlesT;

class GMLS_StokesBoundaryConditions : public BoundaryConditionsT {
    protected:
        typedef Compadre::ParticlesT particle_type;

    public:
        GMLS_StokesBoundaryConditions(Teuchos::RCP<particle_type> particles, mvec_type* b = NULL) :
                BoundaryConditionsT(particles, b) {}

        virtual ~GMLS_StokesBoundaryConditions() {}

        virtual void flagBoundaries();

        virtual void applyBoundaries(local_index_type field_one, local_index_type field_two=-1, scalar_type time=0.0);

        virtual std::vector<InteractingFields> gatherFieldInteractions();
};

}

#endif
