#include <Compadre_AdvectionDiffusion_BoundaryConditions.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;
typedef Compadre::XyzVector xyz_type;

void AdvectionDiffusionBoundaryConditions::flagBoundaries() {
	//device_view_type pts = this->_coords->getPts()->getLocalView<device_view_type>();
	//local_index_type bc_id_size = this->_particles->getFlags()->getLocalLength();
	//Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,bc_id_size), KOKKOS_LAMBDA(const int i) {
	//	scalar_type epsilon_ball = 2e-1;
	//	if (std::abs(pts(i,0)-0.0)<epsilon_ball || std::abs(pts(i,0)-1.0)<epsilon_ball) {
	//	    // left and right
	//	    this->_particles->setFlag(i, 1);
	//	} else if (std::abs(pts(i,1)-0.0)<epsilon_ball || std::abs(pts(i,1)-1.0)<epsilon_ball) {
	//	    // top and bottom
	//	    this->_particles->setFlag(i, 1);
	//	} else {
	//	    this->_particles->setFlag(i, 0);
	//	}
	//});
}

void AdvectionDiffusionBoundaryConditions::applyBoundaries(local_index_type field_one, local_index_type field_two, scalar_type time) {

	//Teuchos::RCP<Compadre::AnalyticFunction> function;
	//function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts(2 /*dimension*/)));


	//TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_b.is_null(), "Tpetra Multivector for BCS not yet specified.");
	//if (field_two == -1) {
	//	field_two = field_one;
	//}

	//host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();
	//host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
	//host_view_type pts = this->_coords->getPts()->getLocalView<host_view_type>();

	//const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	//const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
	//const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

	//for (local_index_type i=0; i<nlocal; ++i) { // parallel_for causes cache thrashing
	//	// get dof corresponding to field
	//	for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
	//		const local_index_type dof = local_to_dof_map(i, field_one, k);
	//		xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
	//		if (bc_id(i,0)==1) rhs_vals(dof,0) = function->evalScalar(pt);
	//	}
	//}

}

std::vector<InteractingFields> AdvectionDiffusionBoundaryConditions::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	//field_interactions.push_back(InteractingFields(op_needing_interaction::bc, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	return field_interactions;
}

}
