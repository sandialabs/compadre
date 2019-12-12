#include "Compadre_BoundaryConditionsT.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"

namespace Compadre {


BoundaryConditionsT::BoundaryConditionsT(	Teuchos::RCP<particle_type> particles,
											mvec_type* b) :
											_coords(particles->getCoordsConst()), _particles(particles), _b(b)
{
	_comm = particles->getCoordsConst()->getComm();
}

}
