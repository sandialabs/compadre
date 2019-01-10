#include <Compadre_AdvectionDiffusion_Sources.hpp>

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

void AdvectionDiffusionSources::evaluateRHS(local_index_type field_one, local_index_type field_two, scalar_type time) {
	Teuchos::RCP<Compadre::AnalyticFunction> function;
    function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts(2 /*dimension*/)));

	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_b.is_null(), "Tpetra Multivector for RHS not yet specified.");
	if (field_two == -1) {
		field_two = field_one;
	}

	// temporary this will be a specific function
	// in the future, we can pass a Teuchos::ArrayRCP of functors
	host_view_type bc_id = this->_particles->getFlags()->getLocalView<host_view_type>();
	host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
	host_view_type pts = this->_coords->getPts()->getLocalView<host_view_type>();

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
	const std::vector<std::vector<std::vector<local_index_type> > >& local_to_dof_map =
			_dof_data->getDOFMap();

	for (local_index_type i=0; i<nlocal; ++i) { // parallel_for causes cache thrashing
		// get dof corresponding to field
		for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
			const local_index_type dof = local_to_dof_map[i][field_one][k];
			xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
			if (bc_id(i,0)==0) rhs_vals(dof,0) = -function->evalScalarLaplacian(pt);
		}
	}
}

std::vector<InteractingFields> AdvectionDiffusionSources::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
    //field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	return field_interactions;
}

}
