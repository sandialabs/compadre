#include <Compadre_GMLS_Staggered_PoissonNeumann_Operator.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_XyzVector.hpp>

#include <Compadre_GMLS.hpp>

#ifdef COMPADREHARNESS_USE_OPENMP
#include <omp.h>
#endif

/*
   Constructs the matrix operator
*/
namespace Compadre{

typedef Compadre::CoordsT coords_type;
typedef Compadre::FieldT fields_type;
typedef Compadre::NeighborhoodT neighborhood_type;
typedef Compadre::XyzVector xyz_type;

void GMLS_Staggered_PoissonNeumannPhysics::initialize() {

    const local_index_type neighbors_needed = GMLS::getNP(Porder);

    bool use_physical_coords = true; // can be set on the operator in the future

    // Loop over all particles, convert to GMLS data types, solve problem and insert into matrix

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type>>& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const neighborhood_type* neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
    const host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();
    const host_view_type nx_vectors = this->_particles->getFieldManager()->getFieldByName("nx")->getMultiVectorPtrConst()->getLocalView<host_view_type>();
    const host_view_type ny_vectors = this->_particles->getFieldManager()->getFieldByName("ny")->getMultiVectorPtrConst()->getLocalView<host_view_type>();
    const host_view_type nz_vectors = this->_particles->getFieldManager()->getFieldByName("nz")->getMultiVectorPtrConst()->getLocalView<host_view_type>();

    //****************
    //
    //  Copying data from particles (std::vector's, multivectors, etc...) to views used by local reconstruction class
    //
    //****************

    // generate the interpolation operator and call the coefficients needed (storing them)
    const coords_type* target_coords = this->_coords;
    const coords_type* source_coords = this->_coords;

    size_t max_num_neighbors = neighborhood->computeMaxNumNeighbors(false /* local processor max*/);

    Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), max_num_neighbors+1);
    Kokkos::View<int**>::HostMirror kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        const int num_i_neighbors = neighborhood->getNumNeighbors(i);
        for (int j=1; j<num_i_neighbors+1; ++j) {
            kokkos_neighbor_lists_host(i, j) = neighborhood->getNeighbor(i,j-1);
        }
        kokkos_neighbor_lists_host(i, 0) = num_i_neighbors;
    });

    Kokkos::View<double**> kokkos_augmented_source_coordinates("source_coordinates", source_coords->nLocal(true /* include halo in count */), source_coords->nDim());
    Kokkos::View<double**>::HostMirror kokkos_augmented_source_coordinates_host = Kokkos::create_mirror_view(kokkos_augmented_source_coordinates);

    // fill in the source coords, adding regular with halo coordiantes into a kokkos view
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int i) {
        xyz_type coordinate = source_coords->getLocalCoords(i, true /*include halo*/, use_physical_coords);
        kokkos_augmented_source_coordinates_host(i,0) = coordinate.x;
        kokkos_augmented_source_coordinates_host(i,1) = coordinate.y;
        kokkos_augmented_source_coordinates_host(i,2) = coordinate.z;
    });

    Kokkos::View<double**> kokkos_target_coordinates("target_coordinates", target_coords->nLocal(), target_coords->nDim());
    Kokkos::View<double**>::HostMirror kokkos_target_coordinates_host = Kokkos::create_mirror_view(kokkos_target_coordinates);
    // fill in the target, adding regular coordiantes only into a kokkos view
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        xyz_type coordinate = target_coords->getLocalCoords(i, false /*include halo*/, use_physical_coords);
        kokkos_target_coordinates_host(i,0) = coordinate.x;
        kokkos_target_coordinates_host(i,1) = coordinate.y;
        kokkos_target_coordinates_host(i,2) = coordinate.z;
    });

    auto epsilons = neighborhood->getHSupportSizes()->getLocalView<const host_view_type>();
    Kokkos::View<double*> kokkos_epsilons("target_coordinates", target_coords->nLocal());
    Kokkos::View<double*>::HostMirror kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        kokkos_epsilons_host(i) = epsilons(i,0);
    });

    Kokkos::View<int*> kokkos_flags("target_flags", target_coords->nLocal());
    Kokkos::View<int*>::HostMirror kokkos_flags_host = Kokkos::create_mirror_view(kokkos_flags);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        kokkos_flags_host(i) = bc_id(i, 0);
    });

    Kokkos::View<double**> kokkos_normals("target_normals", target_coords->nLocal(), target_coords->nDim());
    Kokkos::View<double**>::HostMirror kokkos_normals_host = Kokkos::create_mirror_view(kokkos_normals);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        kokkos_normals_host(i, 0) = nx_vectors(i, 0);
        kokkos_normals_host(i, 1) = ny_vectors(i, 0);
        kokkos_normals_host(i, 2) = nz_vectors(i, 0);
    });

    // Extract out points without any BC
    _noconstraint_filtered_flags = filterViewByID<Kokkos::HostSpace>(kokkos_flags_host, 0);
    auto noconstraint_kokkos_target_coordinates_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_target_coordinates_host,
            _noconstraint_filtered_flags);
    auto noconstraint_kokkos_neighbor_lists_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_neighbor_lists_host,
            _noconstraint_filtered_flags);
    auto noconstraint_kokkos_epsilons_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_epsilons_host,
            _noconstraint_filtered_flags);

    // Extract out points labeled with Dirichlet BC
    _dirichlet_filtered_flags = filterViewByID<Kokkos::HostSpace>(kokkos_flags_host, 1);

    // Extract out points labeled with Neumann BC
    // _neuman_filtered_flags belongs to physics class
    _neumann_filtered_flags = filterViewByID<Kokkos::HostSpace>(kokkos_flags_host, 2);
    auto neumann_kokkos_target_coordinates_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_target_coordinates_host,
            _neumann_filtered_flags);
    auto neumann_kokkos_neighbor_lists_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_neighbor_lists_host,
            _neumann_filtered_flags);
    auto neumann_kokkos_epsilons_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_epsilons_host,
            _neumann_filtered_flags);
    auto neumann_kokkos_normals_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_normals_host,
            _neumann_filtered_flags);

    // Now create a tangent bundles to set for the points with Neumann BC
    int nlocal_neumann = _neumann_filtered_flags.extent(0);
    Kokkos::View<double***> neumann_kokkos_tangent_bundles_host("target_tangent_bundles", nlocal_neumann, target_coords->nDim(), target_coords->nDim());
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal_neumann), KOKKOS_LAMBDA(const int i) {
        neumann_kokkos_tangent_bundles_host(i, 0, 0) = 0.0;
        neumann_kokkos_tangent_bundles_host(i, 1, 0) = 0.0;
        neumann_kokkos_tangent_bundles_host(i, 1, 1) = 0.0;
        neumann_kokkos_tangent_bundles_host(i, 1, 2) = 0.0;
        neumann_kokkos_tangent_bundles_host(i, 2, 0) = neumann_kokkos_normals_host(i, 0);
        neumann_kokkos_tangent_bundles_host(i, 2, 1) = neumann_kokkos_normals_host(i, 1);
        neumann_kokkos_tangent_bundles_host(i, 2, 2) = neumann_kokkos_normals_host(i, 2);
    });

    //****************
    //
    // End of data copying
    //
    //****************

    // No-constraint GMLS operator
    _noconstraint_GMLS = Teuchos::rcp<GMLS>(new GMLS(ReconstructionSpace::VectorTaylorPolynomial,
                        StaggeredEdgeIntegralSample,
                        StaggeredEdgeAnalyticGradientIntegralSample,
                        _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                        3, "SVD", "STANDARD", "NO_CONSTRAINT"));
    _noconstraint_GMLS->setProblemData(noconstraint_kokkos_neighbor_lists_host,
                           kokkos_augmented_source_coordinates_host,
                           noconstraint_kokkos_target_coordinates_host,
                           noconstraint_kokkos_epsilons_host);
    _noconstraint_GMLS->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
    _noconstraint_GMLS->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
    _noconstraint_GMLS->setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
    _noconstraint_GMLS->setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
    _noconstraint_GMLS->setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));
    _noconstraint_GMLS->addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
    _noconstraint_GMLS->generateAlphas(); // just point evaluations

    // Neumann GMLS operator - member of the physics class
    // _neumann_GMLS = Teuchos::rcp<GMLS>(new GMLS(ReconstructionSpace::VectorTaylorPolynomial,
    //                     StaggeredEdgeIntegralSample,
    //                     StaggeredEdgeAnalyticGradientIntegralSample,
    //                     _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
    //                     3, "LU", "STANDARD", "NEUMANN_GRAD_SCALAR"));
    _neumann_GMLS = Teuchos::rcp<GMLS>(new GMLS(ReconstructionSpace::ScalarTaylorPolynomial,
                        StaggeredEdgeAnalyticGradientIntegralSample,
                        _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                        3, "SVD", "STANDARD", "NEUMANN_GRAD_SCALAR"));
    _neumann_GMLS->setProblemData(neumann_kokkos_neighbor_lists_host,
                          kokkos_augmented_source_coordinates_host,
                          neumann_kokkos_target_coordinates_host,
                          neumann_kokkos_epsilons_host);
    _neumann_GMLS->setTangentBundle(neumann_kokkos_tangent_bundles_host);
    _neumann_GMLS->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
    _neumann_GMLS->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
    _neumann_GMLS->setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
    _neumann_GMLS->setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
    _neumann_GMLS->setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));
    _neumann_GMLS->addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
    _neumann_GMLS->generateAlphas(); // just point evaluations

    Teuchos::RCP<Teuchos::Time> GMLSTime = Teuchos::TimeMonitor::getNewCounter ("GMLS");
}

Teuchos::RCP<crs_graph_type> GMLS_Staggered_PoissonNeumannPhysics::computeGraph(local_index_type field_one, local_index_type field_two) {
    if (field_two == -1) {
        field_two = field_one;
    }

    Teuchos::RCP<Teuchos::Time> ComputeGraphTime = Teuchos::TimeMonitor::getNewCounter("Compute Graph Time");
    ComputeGraphTime->start();
    TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A_graph.is_null(), "Tpetra CrsGraph for Physics not yet specified.");

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type>>& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const neighborhood_type* neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    for (local_index_type i=0; i<nlocal; i++) {
        local_index_type num_neighbors = neighborhood->getNumNeighbors(i);
        for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
            local_index_type row = local_to_dof_map(i, field_one, k);

            Teuchos::Array<local_index_type> col_data(num_neighbors*fields[field_two]->nDim());
            Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

            for (local_index_type l=0; l<num_neighbors; l++) {
                for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                    col_data[l*fields[field_two]->nDim() + n] =
                        local_to_dof_map(neighborhood->getNeighbor(i,l), field_two, n);
                }
            }
            {
                this->_A_graph->insertLocalIndices(row, cols);
            }
        }
    }
    ComputeGraphTime->stop();

    return this->_A_graph;
}

void GMLS_Staggered_PoissonNeumannPhysics::computeMatrix(local_index_type field_one, local_index_type field_two, scalar_type time) {

    bool use_physical_coords = true; // can be set on the operator in the future

    Teuchos::RCP<Teuchos::Time> ComputeMatrixTime = Teuchos::TimeMonitor::getNewCounter("Compute Matrix Time");
    ComputeMatrixTime->start();

    TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A.is_null(), "Tpetra CrsMatrix for Physics not yet specified.");

    const local_index_type neighbors_needed = GMLS::getNP(Porder);

    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    const std::vector<Teuchos::RCP<fields_type>>& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const neighborhood_type* neighborhood = this->_particles->getNeighborhoodConst();
    // generate the interpolation operator and call the coefficients needed (storing them)
    const coords_type* target_coords = this->_coords;
    const coords_type* source_coords = this->_coords;

    size_t max_num_neighbors = neighborhood->computeMaxNumNeighbors(false /*local processor max*/);

    // get maximum number of neighbors * fields[field_two]->nDim()
    int team_scratch_size = host_scratch_vector_scalar_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // values
    team_scratch_size += host_scratch_vector_local_index_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // local column indices
    const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently

    // Put values from no-constraint GMLS into matrix
    int nlocal_noconstraint = _noconstraint_filtered_flags.extent(0);
    Kokkos::parallel_for(host_team_policy(nlocal_noconstraint, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
        const int i = teamMember.league_rank();

        host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
        host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

        const local_index_type num_neighbors = neighborhood->getNumNeighbors(_noconstraint_filtered_flags(i));

        // Print error if there's not enough neighbors:
        TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
            "ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed) );

        //Put the values of alpha in the proper place in the global matrix
        for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
            local_index_type row = local_to_dof_map(_noconstraint_filtered_flags(i), field_one, k);
            for (local_index_type l = 0; l < num_neighbors; l++) {
                for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                    col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map(neighborhood->getNeighbor(_noconstraint_filtered_flags(i),l), field_two, n);
                    if (n==k) { // same field, same component
                        val_data(l*fields[field_two]->nDim() + n) = _noconstraint_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l)*_noconstraint_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, false, 0, 0);
                        val_data(0) += _noconstraint_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l)*_noconstraint_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0, 0);
                    } else {
                        val_data(l*fields[field_two]->nDim() + n) = 0.0;
                    }
                }
            }
            {
                this->_A->sumIntoLocalValues(row, num_neighbors * fields[field_two]->nDim(), val_data.data(), col_data.data());//, /*atomics*/false);
            }
        }
    });

    // Put values from neumann GMLS into matrix
    int nlocal_neumann = _neumann_filtered_flags.extent(0);
    Kokkos::parallel_for(host_team_policy(nlocal_neumann, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
        const int i = teamMember.league_rank();

        host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
        host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

        const local_index_type num_neighbors = neighborhood->getNumNeighbors(_neumann_filtered_flags(i));

        // Print error if there's not enough neighbors:
        TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
            "ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed) );

        //Put the values of alpha in the proper place in the global matrix
        for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
            local_index_type row = local_to_dof_map(_neumann_filtered_flags(i), field_one, k);
            for (local_index_type l = 0; l < num_neighbors; l++) {
                for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                    col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map(neighborhood->getNeighbor(_neumann_filtered_flags(i),l), field_two, n);
                    if (n==k) { // same field, same component
                        val_data(l*fields[field_two]->nDim() + n) = _neumann_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l)*_neumann_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, false, 0, 0);
                        val_data(0) += _neumann_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l)*_neumann_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0, 0);
                    } else {
                        val_data(l*fields[field_two]->nDim() + n) = 0.0;
                    }
                }
            }
            {
                this->_A->sumIntoLocalValues(row, num_neighbors * fields[field_two]->nDim(), val_data.data(), col_data.data());//, /*atomics*/false);
            }
        }
    });

    // Apply Dirichlet conditions
    int nlocal_dirichlet = _dirichlet_filtered_flags.extent(0);
    Kokkos::parallel_for(host_team_policy(nlocal_dirichlet, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
        const int i = teamMember.league_rank();

        host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
        host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

        const local_index_type num_neighbors = neighborhood->getNumNeighbors(_dirichlet_filtered_flags(i));

        // Print error if there's not enough neighbors:
        TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
            "ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed) );

        //Put the values of alpha in the proper place in the global matrix
        for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
            local_index_type row = local_to_dof_map(_dirichlet_filtered_flags(i), field_one, k);
            for (local_index_type l = 0; l < num_neighbors; l++) {
                for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                    col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map(neighborhood->getNeighbor(_dirichlet_filtered_flags(i),l), field_two, n);
                    if (n==k) { // same field, same component
                        if (_dirichlet_filtered_flags(i)==neighborhood->getNeighbor(_dirichlet_filtered_flags(i),l)) {
                            val_data(l*fields[field_two]->nDim() + n) = 1.0;
                        } else {
                            val_data(l*fields[field_two]->nDim() + n) = 0.0;
                        }
                    } else {
                        val_data(l*fields[field_two]->nDim() + n) = 0.0;
                    }
                }
            }
            {
                this->_A->sumIntoLocalValues(row, num_neighbors * fields[field_two]->nDim(), val_data.data(), col_data.data());//, /*atomics*/false);
            }
        }
    });
    TEUCHOS_ASSERT(!this->_A.is_null());
    this->_A->print(std::cout);
    ComputeMatrixTime->stop();
}

const std::vector<InteractingFields> GMLS_Staggered_PoissonNeumannPhysics::gatherFieldInteractions() {
    std::vector<InteractingFields> field_interactions;
    field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("neumann solution")));
    return field_interactions;
}

}
