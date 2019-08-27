#include <Compadre_LaplaceBeltrami_BoundaryConditions.hpp>

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

void LaplaceBeltramiBoundaryConditions::flagBoundaries() {
	device_view_type pts = this->_coords->getPts()->getLocalView<device_view_type>();
	local_index_type bc_id_size = this->_particles->getFlags()->getLocalLength();
	if (_parameters->get<local_index_type>("physics number")<3) {
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,bc_id_size), KOKKOS_LAMBDA(const int i) {
		this->_particles->setFlag(i, 0);
	});
	}
}

void LaplaceBeltramiBoundaryConditions::applyBoundaries(local_index_type field_one, local_index_type field_two, scalar_type time) {
	Teuchos::RCP<Compadre::AnalyticFunction> function;
//	function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ConstantEachDimension(1,1,1)));
	if (_parameters->get<local_index_type>("physics number")<3) {
		function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SphereHarmonic(4,5)));
	} else if (_parameters->get<local_index_type>("physics number")==3) {
		function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::FiveStripOnSphere));
	} else if (_parameters->get<local_index_type>("physics number")==10) {
		function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::CylinderSinLonCosZ));
	}

	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_b.is_null(), "Tpetra Multivector for BCS not yet specified.");
	if (field_two == -1) {
		field_two = field_one;
	}

	host_view_type bc_id = this->_particles->getFlags()->getLocalView<host_view_type>();
	host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
	host_view_type pts = this->_coords->getPts()->getLocalView<host_view_type>();


	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<std::vector<std::vector<local_index_type> > >& local_to_dof_map =
			_dof_data->getDOFMap();

	if (field_one == _particles->getFieldManagerConst()->getIDOfFieldFromName("solution") && field_two == _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")) {

		for (local_index_type i=0; i<nlocal; ++i) { // parallel_for causes cache thrashing
			// get dof corresponding to field

			const local_index_type components_of_field_for_rhs = 1;
			for (local_index_type k = 0; k < components_of_field_for_rhs; ++k) {
				const local_index_type dof = local_to_dof_map[i][field_one][k];
				xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
	//			if (bc_id(i,0)==0) rhs_vals(dof,0) = function->evalScalar(pt);
	//			if (bc_id(i,0)==1) rhs_vals(dof,0) = 1;//function->evalScalar(pt);
	//			if (bc_id(i,0)==2) rhs_vals(dof,0) = 0;//function->evalScalar(pt);
				if (bc_id(i,0)==1 || bc_id(i,0)==2) {
					if (_parameters->get<local_index_type>("physics number")<3) {
						rhs_vals(dof,0) = 1.0 / (5 * (5 + 1)) * function->evalScalar(pt);
					} else {
						rhs_vals(dof,0) = function->evalScalar(pt);
					}
				}

	//			if (std::abs((1.0-pts(i,0))*(1.0-pts(i,0)))+std::abs((0.0-pts(i,1))*(0.0+pts(i,1)))+std::abs((0.0-pts(i,2))*(0.0+pts(i,2)))<1.0e-1) {
	//				rhs_vals(dof,0) = 1e-6;
	//			} else {
	//				this->_particles->setFlag(i, 0);
	//			}
			}
		}
	}
}

std::vector<InteractingFields> LaplaceBeltramiBoundaryConditions::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
//	field_interactions.push_back(InteractingFields(op_needing_interaction::bc, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	return field_interactions;
}

}
