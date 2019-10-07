#ifndef _COMPADRE_ADVECTIONDIFFUSIONSOURCES_HPP_
#define _COMPADRE_ADVECTIONDIFFUSIONSOURCES_HPP_

#include <Compadre_SourcesT.hpp>
#include <Compadre_XyzVector.hpp>

namespace Compadre {

class AdvectionDiffusionSources : public SourcesT {

	protected:

		typedef Compadre::ParticlesT particle_type;
        typedef Compadre::XyzVector xyz_type;

        xyz_type _advection_field;
        scalar_type _diffusion;

	public:

		AdvectionDiffusionSources(	Teuchos::RCP<particle_type> particles,
										Teuchos::RCP<mvec_type> b = Teuchos::null) :
										SourcesT(particles, b) {};

		virtual ~AdvectionDiffusionSources() {};

		virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();

        void setAdvectionField(const xyz_type advection_field) {
            for (local_index_type i=0; i<3; ++i) {
                _advection_field[i] = advection_field[i];
            }
        }
        
        xyz_type getAdvectionField() const { return _advection_field; }

        void setDiffusion(const scalar_type diffusion) {
            _diffusion = diffusion;
        }

        scalar_type getDiffusion() const { return _diffusion; }
};

}

#endif
