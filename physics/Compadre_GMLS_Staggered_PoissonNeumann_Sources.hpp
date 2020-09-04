#ifndef _COMPADRE_GMLS_STAGGERED_POISSONNEUMANNSOURCES_HPP_
#define _COMPADRE_GMLS_STAGGERED_POISSONNEUMANNSOURCES_HPP_

#include <Compadre_SourcesT.hpp>
#include <Compadre_XyzVector.hpp>

namespace Compadre {

class GMLS_Staggered_PoissonNeumannPhysics;

class GMLS_Staggered_PoissonNeumannSources : public SourcesT{
    protected:
        typedef Compadre::ParticlesT particle_type;
        GMLS_Staggered_PoissonNeumannPhysics* _physics;

    public:
        GMLS_Staggered_PoissonNeumannSources(Teuchos::RCP<particle_type> particles,
             mvec_type* b = NULL) : SourcesT(particles, b) {};

        virtual ~GMLS_Staggered_PoissonNeumannSources() {};

        virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

        virtual std::vector<InteractingFields> gatherFieldInteractions();

        void setPhysics(Teuchos::RCP<GMLS_Staggered_PoissonNeumannPhysics> physics) { _physics = physics.getRawPtr(); }
};

}
#endif
