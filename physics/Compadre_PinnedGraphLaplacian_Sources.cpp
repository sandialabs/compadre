#include <Compadre_PinnedGraphLaplacian_Sources.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_DOFManager.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;

void PinnedGraphLaplacianSources::evaluateRHS(local_index_type field_one, local_index_type field_two, scalar_type time) {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_b.is_null(), "Tpetra Multivector for RHS not yet specified.");
	if (field_two == -1) {
		field_two = field_one;
	}

	// this function is field independent, and only particle dependent

	// temporary this will be a specific function
	// in the future, we can pass a Teuchos::ArrayRCP of functors
	host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();
	host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

	for (local_index_type i=0; i<nlocal; ++i) { // parallel_for causes cache thrashing
		// get dof corresponding to field
		if (bc_id(i,0)==0) { // bc_id is by particle
			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
				const local_index_type dof = local_to_dof_map(i, field_one, k);
				rhs_vals(dof,0) = 0.0;
			}
		}
	}

}

std::vector<InteractingFields> PinnedGraphLaplacianSources::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::source, 0));
	field_interactions.push_back(InteractingFields(op_needing_interaction::source, 1));
	field_interactions.push_back(InteractingFields(op_needing_interaction::source, 2));
	field_interactions.push_back(InteractingFields(op_needing_interaction::source, 3));
	field_interactions.push_back(InteractingFields(op_needing_interaction::source, 4));
	return field_interactions;
}

}
