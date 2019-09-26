#include <Compadre_PinnedGraphLaplacian_BoundaryConditions.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;

void PinnedGraphLaplacianBoundaryConditions::flagBoundaries() {
	device_view_type pts = this->_coords->getPts()->getLocalView<device_view_type>();
	local_index_type bc_id_size = this->_particles->getFlags()->getLocalLength();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,bc_id_size), KOKKOS_LAMBDA(const int i) {
		if (std::abs((1.0-pts(i,0))*(1.0+pts(i,0)))+std::abs((1.0-pts(i,1))*(1.0+pts(i,1)))+std::abs((1.0-pts(i,2))*(1.0+pts(i,2)))<1.0e-1) {
			this->_particles->setFlag(i, 1);
		} else {
			this->_particles->setFlag(i, 0);
		}
	  });

}

void PinnedGraphLaplacianBoundaryConditions::applyBoundaries(local_index_type field_one, local_index_type field_two, scalar_type time) {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_b.is_null(), "Tpetra Multivector for BCS not yet specified.");
	if (field_two == -1) {
		field_two = field_one;
	}
	host_view_type bc_id = this->_particles->getFlags()->getLocalView<host_view_type>();
	host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

	for (local_index_type i=0; i<nlocal; ++i) { // parallel_for causes cache thrashing
		// get dof corresponding to field
		for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
			const local_index_type dof = local_to_dof_map(i, field_one, k);
			if (bc_id(i,0)==1) rhs_vals(dof,0) = 50.0;
			if (bc_id(i,0)==2) rhs_vals(dof,0) = 0.0;
		}
	}

}

std::vector<InteractingFields> PinnedGraphLaplacianBoundaryConditions::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::bc, 0));
	field_interactions.push_back(InteractingFields(op_needing_interaction::bc, 1));
	field_interactions.push_back(InteractingFields(op_needing_interaction::bc, 2));
	field_interactions.push_back(InteractingFields(op_needing_interaction::bc, 3));
	field_interactions.push_back(InteractingFields(op_needing_interaction::bc, 4));
	return field_interactions;
}

}
