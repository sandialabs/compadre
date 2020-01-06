#ifndef _COMPADRE_GMLS_STOKESSOURCES_HPP_
#define _COMPADRE_GMLS_STOKESSOURCES_HPP_

#include <Compadre_SourcesT.hpp>
#include <Compadre_XyzVector.hpp>

namespace Compadre {

class GMLS_StokesPhysics;

class GMLS_StokesSources : public SourcesT{
    protected:
        typedef Compadre::ParticlesT particle_type;
        GMLS_StokesPhysics* _physics;

    public:
        GMLS_StokesSources(Teuchos::RCP<particle_type> particles, mvec_type* b = NULL) :
                SourcesT(particles, b) {};

        virtual ~GMLS_StokesSources() {};

        virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

        virtual std::vector<InteractingFields> gatherFieldInteractions();

        void setPhysics(Teuchos::RCP<GMLS_StokesPhysics> physics) { _physics = physics.getRawPtr();}
};

}

#endif
