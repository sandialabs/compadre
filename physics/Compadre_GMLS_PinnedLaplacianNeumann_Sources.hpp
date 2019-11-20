#ifndef _COMPADRE_GMLS_PINNEDLAPLACIANNEUMANNSOURCES_HPP_
#define _COMPADRE_GMLS_PINNEDLAPLACIANNEUMANNSOURCES_HPP_

#include <Compadre_SourcesT.hpp>
#include <Compadre_XyzVector.hpp>

namespace Compadre {

class GMLS_PinnedLaplacianNeumannPhysics;

class GMLS_PinnedLaplacianNeumannSources : public SourcesT{
    protected:
        typedef Compadre::ParticlesT particle_type;
        GMLS_PinnedLaplacianNeumannPhysics* _physics;

    public:
        GMLS_PinnedLaplacianNeumannSources(Teuchos::RCP<particle_type> particles,
             Teuchos::RCP<mvec_type> b = Teuchos::null) : SourcesT(particles, b) {};

        virtual ~GMLS_PinnedLaplacianNeumannSources() {};

        virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

        virtual std::vector<InteractingFields> gatherFieldInteractions();

        void setPhysics(Teuchos::RCP<GMLS_PinnedLaplacianNeumannPhysics> physics) { _physics = physics.getRawPtr(); }
};

}
#endif
