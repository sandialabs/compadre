#include <Compadre_ExplicitTimeDependent_Sources.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_ExplicitTimeDependent_Operator.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;
typedef Compadre::XyzVector xyz_type;

void ExplicitTimeDependentSources::evaluateRHS(local_index_type field_one, local_index_type field_two, scalar_type time, scalar_type current_timestep_size, scalar_type previous_timestep_size) {

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

    //if (field_one == _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity")) {
	//Teuchos::RCP<Compadre::AnalyticFunction> function;
	////function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ConstantEachDimension(1,2,3)));
    //function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SinT()));

	TEUCHOS_TEST_FOR_EXCEPT_MSG(_b==NULL, "Tpetra Multivector for RHS not yet specified.");
	if (field_two == -1) {
		field_two = field_one;
	}

	// temporary this will be a specific function
	// in the future, we can pass a Teuchos::ArrayRCP of functors
	auto bc_id = this->_particles->getFlags()->getLocalViewHost(Tpetra::Access::ReadOnly);
	auto rhs_vals = this->_b->getLocalViewHost(Tpetra::Access::OverwriteAll);
	auto pts = this->_coords->getPts()->getLocalViewHost(Tpetra::Access::ReadOnly);

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

	for (local_index_type i=0; i<nlocal; ++i) { // parallel_for causes cache thrashing
		// get dof corresponding to field
		if (bc_id(i,0)==0) {
			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
                const local_index_type dof = local_to_dof_map(i, field_one, k);
				xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
				//rhs_vals(dof,0) = function->evalScalar(pt);
                if (is_velocity) {
                    rhs_vals(dof,0) = _physics->vel_source_function->evalScalar(pt, 0, time);
                } else if (is_height) {
                    rhs_vals(dof,0) = _physics->h_source_function->evalScalar(pt, 0, time);
                } else if (is_d) {
                    rhs_vals(dof,0) = _physics->vel_source_function->evalScalar(pt, 0, time);
                    //printf("ppp(%f): %.16f\n",time,_physics->vel_source_function->evalScalar(pt, 0, time));
                }
				//xyz_type bdry_val = function->evalVector(pt);
				//rhs_vals(dof,0) = bdry_val[k]*time*time*time; // just for testing purposes
			}
		}
	}
    //}
}

std::vector<InteractingFields> ExplicitTimeDependentSources::gatherFieldInteractions() {
    auto timestepper_type = _parameters->get<Teuchos::ParameterList>("time").get<std::string>("type");
	std::vector<InteractingFields> field_interactions;
    if (timestepper_type=="rk") {
	    field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity")));
	    field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getIDOfFieldFromName("height")));
    } else if (timestepper_type=="newmark") {
	    field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getIDOfFieldFromName("d")));
    }
//	field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getField("height")));
//	printf("src: %d %d\n", _particles->getFieldManagerConst()->getField("velocity"), _particles->getFieldManagerConst()->getField("height"));
	return field_interactions;
}

}
