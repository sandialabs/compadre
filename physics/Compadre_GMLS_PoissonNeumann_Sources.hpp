#ifndef _COMPADRE_GMLS_POISSONNEUMANNSOURCES_HPP_
#define _COMPADRE_GMLS_POISSONNEUMANNSOURCES_HPP_

#include <Compadre_SourcesT.hpp>
#include <Compadre_XyzVector.hpp>

namespace Compadre {

class GMLS_PoissonNeumannPhysics;

class GMLS_PoissonNeumannSources : public SourcesT{
    protected:
        typedef Compadre::ParticlesT particle_type;
        GMLS_PoissonNeumannPhysics* _physics;

    public:
        GMLS_PoissonNeumannSources(Teuchos::RCP<particle_type> particles,
             Teuchos::RCP<mvec_type> b = Teuchos::null) : SourcesT(particles, b) {};

        virtual ~GMLS_PoissonNeumannSources() {};

        virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

        virtual std::vector<InteractingFields> gatherFieldInteractions();

        void setPhysics(Teuchos::RCP<GMLS_PoissonNeumannPhysics> physics) { _physics = physics.getRawPtr(); }
};

}
#endif
