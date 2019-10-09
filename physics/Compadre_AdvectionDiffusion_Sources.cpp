#include <Compadre_AdvectionDiffusion_Sources.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include "Compadre_nanoflannInformation.hpp"
#include "Compadre_nanoflannPointCloudT.hpp"
#include <Compadre_AdvectionDiffusion_Operator.hpp>
#include <Compadre_GMLS.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;
typedef Compadre::XyzVector xyz_type;

void AdvectionDiffusionSources::evaluateRHS(local_index_type field_one, local_index_type field_two, scalar_type time) {
    TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_b.is_null(), "Tpetra Multivector for RHS not yet specified.");
    if (field_two == -1) {
        field_two = field_one;
    }

    auto function = Teuchos::rcp(new Compadre::SineProducts(2 /*dimension*/));

	host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
    const host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();

    const std::vector<std::vector<std::pair<size_t, scalar_type> > >& particles_particles_all_neighbors = _physics->_particles_particles_neighborhood->getAllNeighbors();
    const std::vector<std::vector<std::pair<size_t, scalar_type> > >& particle_cells_all_neighbors = _physics->_particle_cells_neighborhood->getAllNeighbors();
    const std::vector<std::vector<std::pair<size_t, scalar_type> > >& cell_particles_all_neighbors = _physics->_cell_particles_neighborhood->getAllNeighbors();

    // quantities contained on cells (mesh)
    auto quadrature_points = _physics->_cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_weights = _physics->_cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_type = _physics->_cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto unit_normals = _physics->_cells->getFieldManager()->getFieldByName("unit_normal")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto adjacent_elements = _physics->_cells->getFieldManager()->getFieldByName("adjacent_elements")->getMultiVectorPtr()->getLocalView<host_view_type>();

    auto penalty = (_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")+1)*_parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")*_physics->_particles_particles_neighborhood->getMinimumHSupportSize();
    // each element must have the same number of edges
    auto num_edges = adjacent_elements.extent(1);
    auto num_interior_quadrature = 0;
    for (int q=0; q<_physics->_weights_ndim; ++q) {
        if (quadrature_type(0,q)!=1) { // some type of edge quadrature
            num_interior_quadrature = q;
            break;
        }
    }
    auto num_exterior_quadrature_per_edge = (_physics->_weights_ndim - num_interior_quadrature)/num_edges;
    //printf("num interior quadrature: %d\n", num_interior_quadrature);
    //printf("num exterior quadrature (per edge): %d\n", num_exterior_quadrature_per_edge);
    //printf("penalty: %f\n", penalty);

    // loop over cells
    //Kokkos::View<int*, Kokkos::HostSpace> row_seen("has row had dirichlet addition", target_coords->nLocal());
    host_vector_local_index_type row_seen("has row had dirichlet addition", nlocal);
    Kokkos::deep_copy(row_seen,0);

    for (int i=0; i<_physics->_cells->getCoordsConst()->nLocal(); ++i) {
        // get all particle neighbors of cell i
        for (size_t j=0; j<cell_particles_all_neighbors[i].size(); ++j) {
            auto particle_j = cell_particles_all_neighbors[i][j].first;
            //Put the values of alpha in the proper place in the global matrix
            //for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
            local_index_type row = local_to_dof_map(particle_j, field_one, 0 /* component 0*/);
            if (bc_id(row, 0) == 0) {
                // loop over quadrature
                double contribution = 0;
                for (int q=0; q<_physics->_weights_ndim; ++q) {
                    if (quadrature_type(i,q)==1) { // interior
                        // integrating RHS function
                        double v = _physics->_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1);
                        xyz_type pt(quadrature_points(i,2*q),quadrature_points(i,2*q+1),0);
                        contribution += quadrature_weights(i,q) * v * function->evalScalar(pt);
                    } else if (quadrature_type(i,q)==2) { // edge on exterior
                        // DG enforcement of Dirichlet BCs
                        int current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                        int adjacent_cell = adjacent_elements(i,current_edge_num);
                        //printf("%d, %d, %d, %d\n", i, q, current_edge_num, adjacent_cell);
                        if (adjacent_cell >= 0) {
                            printf("SHOULD BE NEGATIVE NUMBER!\n");
                        }
                        // penalties for edges of mass
                        double vr = _physics->_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1);
                        xyz_type pt(quadrature_points(i,2*q),quadrature_points(i,2*q+1),0);
                        contribution += penalty * quadrature_weights(i,q) * vr * function->evalScalar(pt);
                    }
                }
                if (row_seen(row)==1) { // already seen
                    rhs_vals(row,0) += contribution;
                } else {
                    rhs_vals(row,0) = contribution;
                    row_seen(row) = 1;
                }
            }
        }
    }
}

std::vector<InteractingFields> AdvectionDiffusionSources::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
    //field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	return field_interactions;
}

}
