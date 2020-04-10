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

    if (field_one != _physics->_pressure_field_id || field_two != _physics->_pressure_field_id) return;

    // pin one DOF for pressure
    host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    if ((_physics->_st_op || _physics->_mix_le_op) && _physics->_use_pinning) {
        const local_index_type dof = local_to_dof_map(0, field_one, 0);
        rhs_vals(dof,0) = 0;
    }

}

std::vector<InteractingFields> ReactionDiffusionBoundaryConditions::gatherFieldInteractions() {
    std::vector<InteractingFields> field_interactions;
    return field_interactions;
}

}
