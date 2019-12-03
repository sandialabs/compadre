#include "Compadre_PhysicsT.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_NeighborhoodT.hpp"

namespace Compadre {

PhysicsT::PhysicsT(	Teuchos::RCP<particle_type> particles,
					Teuchos::RCP<crs_graph_type> A_graph,
					Teuchos::RCP<crs_matrix_type> A) :
					_coords(particles->getCoordsConst()), _particles(particles), _A_graph(A_graph), _A(A)
{
	_comm = particles->getCoordsConst()->getComm();
}

local_index_type PhysicsT::getMaxNumNeighbors() {
    return _particles->getNeighborhood()->computeMaxNumNeighbors(false /*local processor neighbor max*/);
}

}
