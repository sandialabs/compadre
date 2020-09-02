#include <Compadre_GMLS_Stokes_Sources.hpp>
#include <Compadre_GMLS_Stokes_Operator.hpp>
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

void GMLS_StokesSources::evaluateRHS(local_index_type field_one, local_index_type field_two, scalar_type time) {
    Teuchos::RCP<Compadre::AnalyticFunction> velocity_function, velocity_true_function, pressure_function;
    if (_parameters->get<std::string>("solution type")=="tanh") {
        velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::StokesVelocityTestRHS));
        velocity_true_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::StokesVelocityTest));
        pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::StokesPressureTest));
    } else if (_parameters->get<std::string>("solution type")=="sine") {
        velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::CurlCurlSineTestRHS));
        velocity_true_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::CurlCurlSineTest));
        pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts));
    } else {
        velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::CurlCurlPolyTestRHS));
        velocity_true_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::CurlCurlPolyTest));
        pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SecondOrderBasis));
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

    // obtaint the ID for the velocity and the pressure field
    auto velocity_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity");
    auto pressure_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure");

    const coords_type* source_coords = this->_coords;

    for (local_index_type i=0; i<nlocal; i++) {
        for (local_index_type k=0; k < fields[field_one]->nDim(); k++) {
            const local_index_type dof = local_to_dof_map(i, field_one, k);
            xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
            if (field_one == velocity_field_id && field_two == velocity_field_id) {
                if (bc_id(i, 0) == 0) {
                    rhs_vals(dof, 0) = velocity_function->evalVector(pt)[k] + pressure_function->evalScalarDerivative(pt)[k];
                }
            } else if (field_one == pressure_field_id && field_two == pressure_field_id) {
                rhs_vals(dof, 0) = pressure_function->evalScalarLaplacian(pt);
            }
        }
    }

    if (field_one == pressure_field_id && field_two == pressure_field_id) {
        // Obtain the filtered indices of the boundary
        Kokkos::View<local_index_type*> boundary_filtered_flags = _physics->_boundary_filtered_flags;
        local_index_type num_boundary = boundary_filtered_flags.extent(0);
        // Add the penalty from Neumann BC to the RHS
        for (local_index_type i=0; i<num_boundary; i++) {
            for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
                // get dof corresponding to field
                const local_index_type dof = local_to_dof_map(boundary_filtered_flags(i), field_one, k);
                // get the number of neighbors for that targe
                const local_index_type num_neighbors = neighborhood->getNumNeighbors(boundary_filtered_flags(i));
                // obtain the beta value from the constraint
                scalar_type b_i = _physics->_pressure_neumann_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, num_neighbors);

                // Setting up the constraint value - first obtain the gradient of the function
                xyz_type pt(pts(boundary_filtered_flags(i), 0), pts(boundary_filtered_flags(i), 1), pts(boundary_filtered_flags(i), 2));
                xyz_type force_term = pressure_function->evalScalarDerivative(pt) + velocity_function->evalVector(pt);
                scalar_type g = force_term.x*_physics->_pressure_neumann_GMLS->getTangentBundle(i, 2, 0)
                    + force_term.y*_physics->_pressure_neumann_GMLS->getTangentBundle(i, 2, 1)
                    + force_term.z*_physics->_pressure_neumann_GMLS->getTangentBundle(i, 2, 2);
                // Now move the constraint term to the RHS
                rhs_vals(dof, 0) = rhs_vals(dof, 0) - b_i*g;

                // // Using the alpha to evaluate the curl curl coming from the exact velocity
                // scalar_type curlcurl_fdotn_target_value = 0.0;
                // for (local_index_type l=0; l<num_neighbors; l++) {
                //     scalar_type curlcurl_fdotn_neighbor_value = 0.0;
                //     for (local_index_type n=0; n<3; n++) {
                //         scalar_type collapse_value = 0.0;
                //         for (local_index_type m=0; m<3; m++) {
                //             // Obtain the normal direction
                //             scalar_type normal_comp = _physics->_pressure_neumann_GMLS->getTangentBundle(i, 2, m);
                //             collapse_value += normal_comp*_physics->_velocity_all_GMLS->getAlpha1TensorTo1Tensor(TargetOperation::CurlCurlOfVectorPointEvaluation, boundary_filtered_flags(i), m /* output component */, l, n /* input component */);
                //         }
                //         // Get neighbor pt
                //         local_index_type neighbor_idx = neighborhood->getNeighbor(boundary_filtered_flags(i), l);
                //         xyz_type neighbor_pt = source_coords->getLocalCoords(neighbor_idx, true /*include halo*/, true /*use physical coords*/);
                //         curlcurl_fdotn_neighbor_value += collapse_value*velocity_true_function->evalVector(neighbor_pt)[n];
                //     }
                //     curlcurl_fdotn_target_value += curlcurl_fdotn_neighbor_value;
                // }
                // // Move the constraint term to the RHS
                // rhs_vals(dof, 0) = rhs_vals(dof, 0) - b_i*curlcurl_fdotn_target_value;
            }
        }
    }
}

std::vector<InteractingFields> GMLS_StokesSources::gatherFieldInteractions() {
    std::vector<InteractingFields> field_interactions;
    return field_interactions;
}

}
