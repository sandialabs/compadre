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

void ExplicitTimeDependentBoundaryConditions::applyBoundaries(local_index_type field_one, local_index_type field_two, scalar_type time, scalar_type current_timestep_size, scalar_type previous_timestep_size) {

    bool is_velocity = false, is_height = false, is_d = false;
    try { 
       is_velocity = (field_one == _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity"));
    } catch (...) {
    }
    try { 
        is_height = (field_one == _particles->getFieldManagerConst()->getIDOfFieldFromName("height"));
    } catch (...) {
    }
    try { 
        is_d = (field_one == _particles->getFieldManagerConst()->getIDOfFieldFromName("d"));
    } catch (...) {
    }

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_b==NULL, "Tpetra Multivector for BCS not yet specified.");
    if (field_two == -1) {
    	field_two = field_one;
    }

    auto bc_id = this->_particles->getFlags()->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto rhs_vals = this->_b->getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto pts = this->_coords->getPts()->getLocalViewHost(Tpetra::Access::ReadOnly);


    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    // Dirichlet boundary condition for explicit Newmark is enforced by setting the acceleration
    // appropriately at time t_n in order to get the correct displacement at time t_{n+1}
    // This requires knowledge of timestep sizes and previous velocities and accelerations
    // as well as the current (t_{n+1}) displacements.
    double gamma = 0.0;
    decltype(fields[field_one]->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly)) 
        displacement_data, velocity_data, acceleration_data;

    if (is_d) {
        if (_velocity_id.size() < fields.size()) {
            _velocity_id = std::vector<int>(fields.size(), -1);
            _acceleration_id = std::vector<int>(fields.size(), -1);
        }
        gamma = _parameters->get<Teuchos::ParameterList>("time").get<double>("gamma");
        if (previous_timestep_size<0.0) {
            auto this_src_field_name = _particles->getFieldManagerConst()->getFieldByID(field_one)->getName();
            _velocity_id[field_one] = _particles->getFieldManagerConst()->getIDOfFieldFromName(this_src_field_name+"_velocity");
            _acceleration_id[field_one] = _particles->getFieldManagerConst()->getIDOfFieldFromName(this_src_field_name+"_acceleration");
        }
        displacement_data = fields[field_one]->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
        velocity_data = fields[_velocity_id[field_one]]->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
        acceleration_data = fields[_acceleration_id[field_one]]->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
    }

    for (local_index_type i=0; i<nlocal; ++i) { // parallel_for causes cache thrashing
    	// get dof corresponding to field
    	for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
    		const local_index_type dof = local_to_dof_map(i, field_one, k);
    		xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
    		if (bc_id(i,0)==1) {
                if (is_velocity) {
                    rhs_vals(dof,0) = _physics->vel_boundary_function->evalScalar(pt, 0, time);
                } else if (is_height) {
                    rhs_vals(dof,0) = _physics->h_boundary_function->evalScalar(pt, 0, time);
                } else if (is_d) {
                    // for setting acceleration at initial timestep
                    if (previous_timestep_size<0.0) {
                        rhs_vals(dof,0) = 2.0/(current_timestep_size*current_timestep_size)*(_physics->h_boundary_function->evalScalar(pt, 0, time+current_timestep_size)-displacement_data(dof,0)-current_timestep_size*velocity_data(dof,0));
                        //printf("calc: %.16f::: t: %.16f, ts: %.16f, exd: %.16f, oldd: %.16f, oldv: %.16f\n", attempt, time, current_timestep_size, _physics->h_boundary_function->evalScalar(pt, 0, time+current_timestep_size), displacement_data(dof,0), velocity_data(dof,0));
                    } else {
                        rhs_vals(dof,0) = 1.0/(current_timestep_size*current_timestep_size*(gamma+0.5))*(_physics->h_boundary_function->evalScalar(pt, 0, time+current_timestep_size)-displacement_data(dof,0)-current_timestep_size*velocity_data(dof,0)-current_timestep_size*current_timestep_size*(1-gamma)*acceleration_data(dof,0));
                        //printf("calc: %.16f::: t: %.16f, ts: %.16f, exd: %.16f, oldd: %.16f, oldv: %.16f, olda: %.16f\n", attempt, time, current_timestep_size, _physics->h_boundary_function->evalScalar(pt, 0, time+current_timestep_size), displacement_data(dof,0), velocity_data(dof,0), acceleration_data(dof,0));
                    }
                    // exact acceleration
                    //rhs_vals(dof,0) = _physics->vel_source_function->evalScalar(pt, 0, time);
                }
            }
    	}
    }

}

std::vector<InteractingFields> ExplicitTimeDependentBoundaryConditions::gatherFieldInteractions() {
    auto timestepper_type = _parameters->get<Teuchos::ParameterList>("time").get<std::string>("type");
	std::vector<InteractingFields> field_interactions;
    if (timestepper_type=="rk") {
	    field_interactions.push_back(InteractingFields(op_needing_interaction::bc, _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity")));
	    field_interactions.push_back(InteractingFields(op_needing_interaction::bc, _particles->getFieldManagerConst()->getIDOfFieldFromName("height")));
    } else if (timestepper_type=="newmark") {
	    field_interactions.push_back(InteractingFields(op_needing_interaction::bc, _particles->getFieldManagerConst()->getIDOfFieldFromName("d")));
    }
//	printf("bc: %d %d\n", _particles->getFieldManagerConst()->getField("velocity"), _particles->getFieldManagerConst()->getField("height"));
	return field_interactions;
}

}
