#include <Compadre_ReactionDiffusion_BoundaryConditions.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_ReactionDiffusion_Operator.hpp>

namespace Compadre {

void ReactionDiffusionBoundaryConditions::flagBoundaries() {
}

void ReactionDiffusionBoundaryConditions::applyBoundaries(local_index_type field_one, local_index_type field_two, scalar_type time) {

    if (field_one != _physics->_pressure_field_id) return;

    // pin one DOF for pressure
    host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    for (local_index_type i=0; i<nlocal; ++i) { 
        const local_index_type dof = local_to_dof_map(i, field_one, 0);
        rhs_vals(dof,0) = 0;
        // pins first DOF in pressure to 0
    }

}

std::vector<InteractingFields> ReactionDiffusionBoundaryConditions::gatherFieldInteractions() {
    std::vector<InteractingFields> field_interactions;
    return field_interactions;
}

}
