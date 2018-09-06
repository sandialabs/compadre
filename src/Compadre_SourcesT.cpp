#include "Compadre_SourcesT.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_FieldT.hpp"
#include "Compadre_ParticlesT.hpp"

namespace Compadre {

SourcesT::SourcesT(	Teuchos::RCP<SourcesT::particle_type> particles,
					Teuchos::RCP<mvec_type> b) :
					_coords(particles->getCoordsConst()), _particles(particles), _b(b)
{
	_comm = particles->getCoordsConst()->getComm();
}

}
