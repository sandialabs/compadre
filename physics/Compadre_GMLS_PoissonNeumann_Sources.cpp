#include <Compadre_GMLS_PoissonNeumann_Sources.hpp>
#include <Compadre_GMLS_PoissonNeumann_Operator.hpp>
#include <Compadre_GMLS.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_NeighborhoodT.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;
typedef Compadre::XyzVector xyz_type;
typedef Compadre::NeighborhoodT neighborhood_type;

void GMLS_PoissonNeumannSources::evaluateRHS(local_index_type field_one, local_index_type field_two, scalar_type time, scalar_type current_timestep_size, scalar_type previous_timestep_size) {
    Teuchos::RCP<Compadre::AnalyticFunction> function;
    if (_parameters->get<std::string>("solution type")=="sine") {
        function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts));
    } else {
        function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SecondOrderBasis));
    }

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_b==NULL, "Tpetra Multivector for RHS not yet specified.");
    if (field_two == -1) {
        field_two = field_one;
    }

    host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();
    host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
    host_view_type pts = this->_coords->getPts()->getLocalView<host_view_type>();
    const neighborhood_type* neighborhood = this->_particles->getNeighborhoodConst();

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type>>& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    for (local_index_type i=0; i<nlocal; i++) { // parallel_for causes cache thrashing
        // get dof corresponding to field
        for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
            const local_index_type dof = local_to_dof_map(i, field_one, k);
            xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
            if (bc_id(i, 0) != 1) rhs_vals(dof, 0) = function->evalScalarLaplacian(pt);
        }
    }

    // Obtain the filtered indices of the Neumann
    Kokkos::View<local_index_type*> neumann_filtered_flags = _physics->_neumann_filtered_flags;
    local_index_type num_neumann = neumann_filtered_flags.extent(0);
    // Add the penalty from Neumann BC to the RHS
    for (local_index_type i=0; i<num_neumann; i++) {
        for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
            // get dof corresponding to field
            const local_index_type dof = local_to_dof_map(neumann_filtered_flags(i), field_one, k);

            // get the number of neighbors for that target
            const local_index_type num_neighbors = neighborhood->getNumNeighbors(neumann_filtered_flags(i));
            // obtain the beta value from the constraint
            scalar_type b_i = _physics->_neumann_GMLS->getSolutionSetHost()->getAlpha0TensorTo0Tensor(TargetOperation::LaplacianOfScalarPointEvaluation, i, num_neighbors);

            // Setting up the constraing value - first obtain the gradient of the function
            xyz_type pt(pts(neumann_filtered_flags(i), 0), pts(neumann_filtered_flags(i), 1), pts(neumann_filtered_flags(i), 2));
            xyz_type grad_pt = function->evalScalarDerivative(pt);
            // Then perform the dot product with the normal vector
           scalar_type g = grad_pt.x*_physics->_neumann_GMLS->getTangentBundle(i, 2, 0)
               + grad_pt.y*_physics->_neumann_GMLS->getTangentBundle(i, 2, 1)
               + grad_pt.z*_physics->_neumann_GMLS->getTangentBundle(i, 2, 2);

            // Now move the constraint term to the RHS
            rhs_vals(dof, 0) = rhs_vals(dof, 0) - b_i*g;
        }
    }
}

std::vector<InteractingFields> GMLS_PoissonNeumannSources::gatherFieldInteractions() {
    std::vector<InteractingFields> field_interactions;
    field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getIDOfFieldFromName("neumann solution")));
    return field_interactions;
}

}
