#include <Compadre_LagrangianShallowWater_Sources.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_XyzVector.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;
typedef Compadre::XyzVector xyz_type;

void LagrangianShallowWaterSources::evaluateRHS(local_index_type field_one, local_index_type field_two, scalar_type time, scalar_type current_timestep_size, scalar_type previous_timestep_size) {

    //if (field_one == _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity")) {
	//    Teuchos::RCP<Compadre::AnalyticFunction> function;
	//    function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ConstantEachDimension(1,2,3)));

	//    TEUCHOS_TEST_FOR_EXCEPT_MSG(_b==NULL, "Tpetra Multivector for RHS not yet specified.");
	//    if (field_two == -1) {
	//    	field_two = field_one;
	//    }

	//    // temporary this will be a specific function
	//    // in the future, we can pass a Teuchos::ArrayRCP of functors
	//    host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();
	//    host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
	//    host_view_type pts = this->_coords->getPts()->getLocalView<host_view_type>();

	//    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	//    const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    //    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

	//    for (local_index_type i=0; i<nlocal; ++i) { // parallel_for causes cache thrashing
	//    	// get dof corresponding to field
	//    	if (bc_id(i,0)==0) {
	//    		for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
    //                const local_index_type dof = local_to_dof_map(i, field_one, k);
	//    			xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
	//    			xyz_type bdry_val = function->evalVector(pt);
	//    			rhs_vals(dof,0) = bdry_val[k]*time*time*time; // just for testing purposes
	//    		}
	//    	}
	//    }
    //}
}

std::vector<InteractingFields> LagrangianShallowWaterSources::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
//	field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity")));
//	field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getField("height")));
//	printf("src: %d %d\n", _particles->getFieldManagerConst()->getField("velocity"), _particles->getFieldManagerConst()->getField("height"));
	return field_interactions;
}

}
