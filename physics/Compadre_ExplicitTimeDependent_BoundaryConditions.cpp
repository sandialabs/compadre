#include <Compadre_ExplicitTimeDependent_BoundaryConditions.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_ExplicitTimeDependent_Operator.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;
typedef Compadre::XyzVector xyz_type;

void ExplicitTimeDependentBoundaryConditions::flagBoundaries() {
	//device_view_type pts = this->_coords->getPts()->getLocalView<device_view_type>();
	//local_index_type bc_id_size = this->_particles->getFlags()->getLocalLength();
	//Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,bc_id_size), KOKKOS_LAMBDA(const int i) {
	//	scalar_type epsilon_ball = 1e-6;
	//	if (std::abs(pts(i,0)-1.0)<epsilon_ball || std::abs(pts(i,0)+1.0)<epsilon_ball || std::abs(pts(i,1)-1.0)<epsilon_ball || std::abs(pts(i,1)+1.0)<epsilon_ball || std::abs(pts(i,2)-1.0)<epsilon_ball || std::abs(pts(i,2)+1.0)<epsilon_ball) {
	//		this->_particles->setFlag(i, 1);
	//	} else {
	//		this->_particles->setFlag(i, 0);
	//	}
	//});

}

void ExplicitTimeDependentBoundaryConditions::applyBoundaries(local_index_type field_one, local_index_type field_two, scalar_type time) {

    bool is_velocity = (field_one == _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity"));
    bool is_height = (field_one == _particles->getFieldManagerConst()->getIDOfFieldFromName("height"));

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_b==NULL, "Tpetra Multivector for BCS not yet specified.");
    if (field_two == -1) {
    	field_two = field_one;
    }

    host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();
    host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
    host_view_type pts = this->_coords->getPts()->getLocalView<host_view_type>();


    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    for (local_index_type i=0; i<nlocal; ++i) { // parallel_for causes cache thrashing
    	// get dof corresponding to field
    	for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
    		const local_index_type dof = local_to_dof_map(i, field_one, k);
    		xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
    		if (bc_id(i,0)==1) {
                if (is_velocity) {
                    rhs_vals(dof,0) = _physics->vel_boundary_function->evalScalar(pt, 0, time);
                } else {
                    rhs_vals(dof,0) = _physics->h_boundary_function->evalScalar(pt, 0, time);
                }
            }
    	}
    }

}

std::vector<InteractingFields> ExplicitTimeDependentBoundaryConditions::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::bc, _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity")));
	field_interactions.push_back(InteractingFields(op_needing_interaction::bc, _particles->getFieldManagerConst()->getIDOfFieldFromName("height")));
//	printf("bc: %d %d\n", _particles->getFieldManagerConst()->getField("velocity"), _particles->getFieldManagerConst()->getField("height"));
	return field_interactions;
}

}
