#include <Compadre_GMLS_Stokes_Operator.hpp>

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
Construct the matrix operator
*/

namespace Compadre {

typedef Compadre::CoordsT coords_type;
typedef Compadre::FieldT fields_type;
typedef Compadre::NeighborhoodT neighborhood_type;
typedef Compadre::XyzVector xyz_type;

Kokkos::View<size_t*, Kokkos::HostSpace> GMLS_StokesPhysics::getMaxEntriesPerRow(local_index_type field_one, local_index_type field_two) {

    auto comm = this->_particles->getCoordsConst()->getComm();

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    Kokkos::View<size_t*, Kokkos::HostSpace> maxEntriesPerRow("max entries per row", nlocal);

    // obtain the ID for velocity and pressure field
    auto velocity_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity");
    auto pressure_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure");
    auto lm_pressure_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("lm_pressure");

    const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
    const std::vector<Teuchos::RCP<fields_type>>& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();

    if ( (field_one == velocity_field_id && field_two == velocity_field_id) ||
         (field_one == velocity_field_id && field_two == pressure_field_id) ||
         (field_one == pressure_field_id && field_two == pressure_field_id) ||
         (field_one == pressure_field_id && field_two == velocity_field_id) ) {
        for (local_index_type i=0; i<nlocal; i++) {
            maxEntriesPerRow(i) = fields[field_one]->nDim()*fields[field_two]->nDim()*neighborhood->getNumNeighbors(i);
        }
    } else if (field_one == lm_pressure_field_id && field_two == pressure_field_id) {
        auto row_map_entries = _row_map->getMyGlobalIndices();
        Kokkos::resize(maxEntriesPerRow, row_map_entries.extent(0));
        Kokkos::deep_copy(maxEntriesPerRow, 0);
        if (comm->getRank()==0) {
            maxEntriesPerRow(0) = nlocal*fields[field_two]->nDim();
        } else {
            maxEntriesPerRow(row_map_entries.extent(0)-1) = nlocal*fields[field_two]->nDim();
        }
        // write to last entry
    } else if (field_one == pressure_field_id && field_two == lm_pressure_field_id) {
        for (local_index_type i=0; i<nlocal; i++) {
            maxEntriesPerRow(i) = fields[field_two]->nDim();
        }
    } else if (field_one == lm_pressure_field_id && field_two == lm_pressure_field_id) {
        for (local_index_type i=0; i<nlocal; i++) {
            maxEntriesPerRow(i) = fields[field_two]->nDim();
        }
    }

    return maxEntriesPerRow;
}

void GMLS_StokesPhysics::initialize() {
    const local_index_type neighbors_needed = GMLS::getNP(Porder);

    bool use_physical_coords = true; // can be set on the operator in the future

    // Loop over all particles, convert to GMLS data types and solve GMLS problems

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type>>& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const neighborhood_type* neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
    const host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();
    const host_view_type nx_vectors = this->_particles->getFieldManager()->getFieldByName("nx")->getMultiVectorPtrConst()->getLocalView<host_view_type>();
    const host_view_type ny_vectors = this->_particles->getFieldManager()->getFieldByName("ny")->getMultiVectorPtrConst()->getLocalView<host_view_type>();
    const host_view_type nz_vectors = this->_particles->getFieldManager()->getFieldByName("nz")->getMultiVectorPtrConst()->getLocalView<host_view_type>();

    //***************
    //
    // Copying data from particles to views used by local reconstruction class
    //
    //***************

    const coords_type* target_coords = this->_coords;
    const coords_type* source_coords = this->_coords;

    size_t max_num_neighbors = neighborhood->computeMaxNumNeighbors(false /*local processor max*/);

    Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), max_num_neighbors+1);
    Kokkos::View<int**>::HostMirror kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        const int num_i_neighbors = neighborhood->getNumNeighbors(i);
        for (int j=1; j<num_i_neighbors+1; ++j) {
            kokkos_neighbor_lists_host(i, j) = neighborhood->getNeighbor(i, j-1);
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
    Kokkos::View<double*> kokkos_epsilons("epsilons", target_coords->nLocal());
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
    _internal_filtered_flags = filterViewByID<Kokkos::HostSpace>(kokkos_flags_host, 0);

    // Extract out points labeled with Neumann BC
    _boundary_filtered_flags = filterViewByID<Kokkos::HostSpace>(kokkos_flags_host, 1);
    auto boundary_kokkos_target_coordinates_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_target_coordinates_host,
            _boundary_filtered_flags);
    auto boundary_kokkos_neighbor_lists_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_neighbor_lists_host,
            _boundary_filtered_flags);
    auto boundary_kokkos_epsilons_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_epsilons_host,
            _boundary_filtered_flags);
    auto boundary_kokkos_normals_host = Extract::extractViewByIndex<Kokkos::HostSpace>(kokkos_normals_host,
            _boundary_filtered_flags);

    // Now create a tangent bundles to set for the points with boundary BC
    int nlocal_boundary = _boundary_filtered_flags.extent(0);
    Kokkos::View<double***> boundary_kokkos_tangent_bundles_host("target_tangent_bundles", nlocal_boundary, target_coords->nDim(), target_coords->nDim());
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal_boundary), KOKKOS_LAMBDA(const int i) {
        boundary_kokkos_tangent_bundles_host(i, 0, 0) = 0.0;
        boundary_kokkos_tangent_bundles_host(i, 0, 1) = 0.0;
        boundary_kokkos_tangent_bundles_host(i, 0, 2) = 0.0;
        boundary_kokkos_tangent_bundles_host(i, 1, 0) = 0.0;
        boundary_kokkos_tangent_bundles_host(i, 1, 1) = 0.0;
        boundary_kokkos_tangent_bundles_host(i, 1, 2) = 0.0;
        boundary_kokkos_tangent_bundles_host(i, 2, 0) = boundary_kokkos_normals_host(i, 0);
        boundary_kokkos_tangent_bundles_host(i, 2, 1) = boundary_kokkos_normals_host(i, 1);
        boundary_kokkos_tangent_bundles_host(i, 2, 2) = boundary_kokkos_normals_host(i, 2);
    });

    //****************
    //
    // End of data copying
    //
    //****************

    Teuchos::RCP<Teuchos::Time> GMLSTime = Teuchos::TimeMonitor::getNewCounter("GMLS");
    GMLSTime->start();

    // velocity GMLS operator
    _velocity_all_GMLS = Teuchos::rcp<GMLS>(new GMLS(ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial,
                        VectorPointSample,
                        _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                        3, "SVD", "STANDARD", "NO_CONSTRAINT"));
    _velocity_all_GMLS->setProblemData(kokkos_neighbor_lists_host,
                        kokkos_augmented_source_coordinates_host,
                        kokkos_target_coordinates_host,
                        kokkos_epsilons_host);
    _velocity_all_GMLS->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
    _velocity_all_GMLS->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
    _velocity_all_GMLS->addTargets(TargetOperation::CurlCurlOfVectorPointEvaluation);
    _velocity_all_GMLS->generateAlphas(_parameters->get<Teuchos::ParameterList>("remap").get<int>("number of batches"));

    // pressure GMLS operator - all points
    _pressure_all_GMLS = Teuchos::rcp<GMLS>(new GMLS(ReconstructionSpace::VectorTaylorPolynomial,
                        StaggeredEdgeIntegralSample,
                        StaggeredEdgeAnalyticGradientIntegralSample,
                        _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                        3, "SVD", "STANDARD", "NO_CONSTRAINT"));
    _pressure_all_GMLS->setProblemData(kokkos_neighbor_lists_host,
                        kokkos_augmented_source_coordinates_host,
                        kokkos_target_coordinates_host,
                        kokkos_epsilons_host);
    _pressure_all_GMLS->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
    _pressure_all_GMLS->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
    _pressure_all_GMLS->setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
    _pressure_all_GMLS->setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
    _pressure_all_GMLS->setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));
    _pressure_all_GMLS->addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
    _pressure_all_GMLS->addTargets(TargetOperation::GradientOfScalarPointEvaluation);
    _pressure_all_GMLS->generateAlphas(_parameters->get<Teuchos::ParameterList>("remap").get<int>("number of batches"));

    // pressure GMLS operator - boundary
    _pressure_neumann_GMLS = Teuchos::rcp<GMLS>(new GMLS(ReconstructionSpace::VectorTaylorPolynomial,
                        StaggeredEdgeIntegralSample,
                        StaggeredEdgeAnalyticGradientIntegralSample,
                        _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                        3, "SVD", "STANDARD", "NEUMANN_GRAD_SCALAR"));
    _pressure_neumann_GMLS->setProblemData(boundary_kokkos_neighbor_lists_host,
                        kokkos_augmented_source_coordinates_host,
                        boundary_kokkos_target_coordinates_host,
                        boundary_kokkos_epsilons_host);
    _pressure_neumann_GMLS->setTangentBundle(boundary_kokkos_tangent_bundles_host);
    _pressure_neumann_GMLS->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
    _pressure_neumann_GMLS->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
    _pressure_neumann_GMLS->setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
    _pressure_neumann_GMLS->setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
    _pressure_neumann_GMLS->setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));
    _pressure_neumann_GMLS->addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
    _pressure_neumann_GMLS->generateAlphas(_parameters->get<Teuchos::ParameterList>("remap").get<int>("number of batches"));

    GMLSTime->stop();
}

Teuchos::RCP<crs_graph_type> GMLS_StokesPhysics::computeGraph(local_index_type field_one, local_index_type field_two) {

    if (field_two == -1) {
        field_two = field_one;
    }

    Teuchos::RCP<Teuchos::Time> ComputeGraphTime = Teuchos::TimeMonitor::getNewCounter("Compute Graph Time");
    ComputeGraphTime->start();

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_row_map.is_null(), "Row map used for construction of CrsGraph before being initialized.");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_col_map.is_null(), "Column map used for construction of CrsGraph before being initialized.");

    // obtain the ID for velocity and pressure field
    auto velocity_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity");
    auto pressure_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure");
    auto lm_pressure_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("lm_pressure");

    if (field_one == lm_pressure_field_id && field_two == pressure_field_id) {
        // row all DOFs for solution against Lagrange multiplier

        // create a new graph from the existing one, but with an updated row map augmented by a single global dof
        // for the Lagrange multiplier
        auto existing_row_map = _row_map;
        auto existing_col_map = _col_map;

        size_t max_entries_per_row;
        Teuchos::ArrayRCP<const size_t> empty_array;
        bool bound_same_for_all_local_rows = true;
        this->_A_graph->getNumEntriesPerLocalRowUpperBound(empty_array, max_entries_per_row, bound_same_for_all_local_rows);

        auto row_map_index_base = existing_row_map->getIndexBase();
        auto row_map_entries = existing_row_map->getMyGlobalIndices();

        auto comm = this->_particles->getCoordsConst()->getComm();
        const int offset_size = (comm->getRank() == 0) ? 0 : 1;
        Kokkos::View<global_index_type*> new_row_map_entries("", row_map_entries.extent(0)+offset_size);

        for (size_t i=0; i<row_map_entries.extent(0); i++) {
            new_row_map_entries(i) = row_map_entries(i);
        }

        {
            // exchange information between processors to get first index on processor 0
            global_index_type global_index_proc_0_element_0;
            if (comm->getRank()==0) {
                global_index_proc_0_element_0 = row_map_entries(0);
            }

            // broadcast from processor 0 to other ranks
            Teuchos::broadcast<local_index_type, global_index_type>(*comm, 0 /*processor broadcasting*/,
                    1 /*size*/, &global_index_proc_0_element_0);

            if (comm->getRank() > 0) {
                // Now the first entry in the map is to the shared grid of the Lagrange multiplier
                new_row_map_entries(new_row_map_entries.extent(0)-1) = global_index_proc_0_element_0;
            }
        }

        auto new_row_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                    new_row_map_entries,
                    row_map_index_base,
                    this->_particles->getCoordsConst()->getComm()));

        this->setRowMap(new_row_map);

        auto entries = this->getMaxEntriesPerRow(field_one, field_two);
        auto dual_view_entries = Kokkos::DualView<size_t*>("dual view", entries.extent(0));
        auto host_view_entries = dual_view_entries.h_view;
        Kokkos::deep_copy(host_view_entries, entries);
        dual_view_entries.modify<Kokkos::DefaultHostExecutionSpace>();
        dual_view_entries.sync<Kokkos::DefaultExecutionSpace>();
        _A_graph = Teuchos::rcp(new crs_graph_type (_row_map, _col_map, dual_view_entries, Tpetra::StaticProfile));
    } else if (field_one == pressure_field_id && field_two == lm_pressure_field_id) {
        // create a new graph from the existing one, but with an updated col map augmented by a single global dof
        // for the Lagrange multiplier
        auto existing_row_map = _row_map;
        auto existing_col_map = _col_map;

        size_t max_entries_per_row;
        Teuchos::ArrayRCP<const size_t> empty_array;
        bool bound_same_for_all_local_rows = true;
        this->_A_graph->getNumEntriesPerLocalRowUpperBound(empty_array, max_entries_per_row, bound_same_for_all_local_rows);

        auto col_map_index_base = existing_col_map->getIndexBase();
        auto col_map_entries = existing_col_map->getMyGlobalIndices();
        auto row_map_entries = existing_row_map->getMyGlobalIndices();

        auto comm = this->_particles->getCoordsConst()->getComm();
        const int offset_size = (comm->getRank() == 0) ? 0 : 1;
        Kokkos::View<global_index_type*> new_col_map_entries("", col_map_entries.extent(0)+offset_size);

        for (size_t i=0; i<col_map_entries.extent(0); i++) {
            new_col_map_entries(i) = col_map_entries(i);
        }

        {
            // exchange information between processors to get first index on processor 0
            global_index_type global_index_proc_0_element_0;
            if (comm->getRank()==0) {
                global_index_proc_0_element_0 = col_map_entries(0);
            }

            // broadcast from processor 0 to other ranks
            Teuchos::broadcast<local_index_type, global_index_type>(*comm, 0 /*processor broadcasting*/,
                    1 /*size*/, &global_index_proc_0_element_0);

            if (comm->getRank() > 0) {
                // now the first entry in the map is to the shared grid for the Lagrange multiplier
                new_col_map_entries(new_col_map_entries.extent(0)-1) = global_index_proc_0_element_0;
            }
        }

        auto new_col_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                    new_col_map_entries,
                    col_map_index_base,
                    this->_particles->getCoordsConst()->getComm()));

        this->setColMap(new_col_map);

        auto entries = this->getMaxEntriesPerRow(field_one, field_two);
        auto dual_view_entries = Kokkos::DualView<size_t*>("dual view", entries.extent(0));
        auto host_view_entries = dual_view_entries.h_view;
        Kokkos::deep_copy(host_view_entries, entries);
            dual_view_entries.modify<Kokkos::DefaultHostExecutionSpace>();
        dual_view_entries.sync<Kokkos::DefaultExecutionSpace>();
        _A_graph = Teuchos::rcp(new crs_graph_type (_row_map, _col_map, dual_view_entries, Tpetra::StaticProfile));
    }

    if (_A_graph.is_null()) {
        auto entries = this->getMaxEntriesPerRow(field_one, field_two);
        auto dual_view_entries = Kokkos::DualView<size_t*>("dual_view", entries.extent(0));
        auto host_view_entries = dual_view_entries.h_view;
        Kokkos::deep_copy(host_view_entries, entries);
        dual_view_entries.modify<Kokkos::DefaultHostExecutionSpace>();
        dual_view_entries.sync<Kokkos::DefaultHostExecutionSpace>();
        _A_graph = Teuchos::rcp(new crs_graph_type (_row_map, _col_map, dual_view_entries, Tpetra::StaticProfile));
    }

    if (!_A_graph->isFillActive()) _A_graph->resumeFill();

    // Check that the solver is in 'blocked' mode
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==false,
        "Non-blocked matrix system incompatible with Lagrange multiplier based in Stokes' solve.");

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type>>& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const neighborhood_type* neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
    size_t max_num_neighbors = neighborhood->computeMaxNumNeighbors(false /*local processor max*/);

    if ( (field_one == velocity_field_id && field_two == velocity_field_id) ||
         (field_one == velocity_field_id && field_two == pressure_field_id) ||
         (field_one == pressure_field_id && field_two == pressure_field_id) ||
         (field_one == pressure_field_id && field_two == velocity_field_id) ) {
        for (local_index_type i=0; i<nlocal; i++) {
            local_index_type num_neighbors = neighborhood->getNumNeighbors(i);
            for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
                local_index_type row = local_to_dof_map(i, field_one, k);
                Teuchos::Array<local_index_type> col_data(max_num_neighbors*fields[field_two]->nDim());
                Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

                for (local_index_type l=0; l<num_neighbors; l++) {
                    for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                        col_data[l*fields[field_two]->nDim() +n] =
                            local_to_dof_map(neighborhood->getNeighbor(i, l), field_two, n);
                    }
                }
                {
                    this->_A_graph->insertLocalIndices(row, cols);
                }
            }
        }
    } else if (field_one == lm_pressure_field_id && field_two == pressure_field_id) {
        // row all DOFs for pressure against Lagrange multiplier
        Teuchos::Array<local_index_type> col_data(nlocal*fields[field_two]->nDim());
        Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

        for (local_index_type l=0; l<nlocal; l++) {
            for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                cols[l*fields[field_two]->nDim() + n] = local_to_dof_map(l, field_two, n);
            }
        }

        auto comm = this->_particles->getCoordsConst()->getComm();
        if (comm->getRank() == 0) {
            // local index 0 is the shared global id for the Lagrange multiplier
            this->_A_graph->insertLocalIndices(0, cols);
        } else {
            // local index 0 is the shared global id for the Lagrange multiplier
            this->_A_graph->insertLocalIndices(nlocal, cols);
        }
    } else if (field_one == pressure_field_id && field_two == lm_pressure_field_id) {
        // col all DOFs for pressure agsint Lagrange multiplier
        auto comm = this->_particles->getCoordsConst()->getComm();
        for (local_index_type i=0; i<nlocal; i++) {
            for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
                local_index_type row = local_to_dof_map(i, field_one, k);
                Teuchos::Array<local_index_type> col_data(1);
                Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
                if (comm->getRank() == 0) {
                    // local index 0 is global shared index
                    cols[0] = 0;
                } else {
                    // local index 0 is global shared index
                    cols[0] = nlocal;
                }
                this->_A_graph->insertLocalIndices(row, cols);
            }
        }
    } else if (field_one == lm_pressure_field_id && field_two == lm_pressure_field_id) {
        // Identity on all DOFs for Lagrange multiplier
        for (local_index_type i=0; i<nlocal; i++) {
            for (local_index_type k=0; k<fields[field_two]->nDim(); k++) {
                local_index_type row = local_to_dof_map(i, field_one, k);
                Teuchos::Array<local_index_type> col_data(fields[field_two]->nDim());
                Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

                for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                    cols[n] = local_to_dof_map(i, field_two, n);
                }
                {
                    this->_A_graph->insertLocalIndices(row, cols);
                }
            }
        }
    }

    ComputeGraphTime->stop();

    return this->_A_graph;
}

void GMLS_StokesPhysics::computeMatrix(local_index_type field_one, local_index_type field_two, scalar_type time) {

    bool use_physical_coords = true; // can be set on the operator in the future

    Teuchos::RCP<Teuchos::Time> ComputeMatrixTime = Teuchos::TimeMonitor::getNewCounter("Compute Matrix Time");
    ComputeMatrixTime->start();

    TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A.is_null(), "Tpetra CrsMatrix for Physics not yet specified.");

    // obtain the ID for velocity and pressure field
    auto velocity_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity");
    auto pressure_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure");
    auto lm_pressure_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("lm_pressure");

    const local_index_type neighbors_needed = GMLS::getNP(Porder);

    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    const std::vector<Teuchos::RCP<fields_type>>& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const neighborhood_type* neighborhood = this->_particles->getNeighborhoodConst();

    const coords_type* target_coords = this->_coords;
    const coords_type* source_coords = this->_coords;

    size_t max_num_neighbors = neighborhood->computeMaxNumNeighbors(false /*local processor max*/);
    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();

    // get maximum number of neighbors*fields[field_two]->nDim()
    int team_scratch_size = host_scratch_vector_scalar_type::shmem_size(max_num_neighbors*fields[field_two]->nDim()); // values
    team_scratch_size += host_scratch_vector_local_index_type::shmem_size(max_num_neighbors*fields[field_two]->nDim()); // local column indicies
    const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently

    // Put values from no-constraint GMLS into matrix
    if ( (field_one == velocity_field_id && field_two == velocity_field_id) ||
         (field_one == velocity_field_id && field_two == pressure_field_id) ) {
        Kokkos::parallel_for(host_team_policy(nlocal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level, Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
            const int i = teamMember.league_rank();

            host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
            host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

            const local_index_type num_neighbors = neighborhood->getNumNeighbors(i);

            // Print error if there's not enough neighbors
            TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
                    "ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed));

            // Put the values of alpha in the proper place in the global matrix
            for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
                local_index_type row = local_to_dof_map(i, field_one, k);
                for (local_index_type l=0; l<num_neighbors; l++) {
                    for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                        col_data(l*fields[field_two]->nDim()+n) = local_to_dof_map(neighborhood->getNeighbor(i, l), field_two, n);
                        if (field_one == velocity_field_id && field_two == velocity_field_id) {
                            // checking its boundary condition
                            if (bc_id(i, 0) == 1) {
                                // for points on the boundary
                                if (i==neighborhood->getNeighbor(i, l)) {
                                    if (n==k) {
                                        val_data(l*fields[field_two]->nDim() + n) = 1.0;
                                    } else {
                                        val_data(l*fields[field_two]->nDim() + n) = 0.0;
                                    }
                                } else {
                                    val_data(l*fields[field_two]->nDim() + n) = 0.0;
                                }
                            } else {
                                // for others, evaluate the coefficient and fill the row
                                val_data(l*fields[field_two]->nDim() + n) = _velocity_all_GMLS->getAlpha1TensorTo1Tensor(TargetOperation::CurlCurlOfVectorPointEvaluation, i, k /* output component*/, l, n /*input component*/);
                            }
                        } else if (field_one == velocity_field_id && field_two == pressure_field_id) {
                            if (bc_id(i, 0) == 1) {
                                val_data(l*fields[field_two]->nDim() + n) = 0.0;
                            } else {
                                val_data(l*fields[field_two]->nDim() + n) = _pressure_all_GMLS->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, k, l)*_pressure_all_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample,i, l, false, 0, 0);
                                val_data(0) += _pressure_all_GMLS->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, k, l)*_pressure_all_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0, 0);
                            }
                        }
                    }
                }
                {
                    this->_A->sumIntoLocalValues(row, num_neighbors*fields[field_two]->nDim(), val_data.data(), col_data.data());
                }
            }
        });
    } else if (field_one == pressure_field_id && field_two == velocity_field_id) {
        // Put values into pressure-velocity block

        // Put values from Neumann GMLS into matrix
        int nlocal_boundary = _boundary_filtered_flags.extent(0);
        Kokkos::parallel_for(host_team_policy(nlocal_boundary, Kokkos::AUTO).set_scratch_size(host_scratch_team_level, Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
            const int i = teamMember.league_rank();
            host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
            host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
            const local_index_type num_neighbors = neighborhood->getNumNeighbors(_boundary_filtered_flags(i));

            // Obtain the value of b_i from Neumann GMLS
            scalar_type b_i = _pressure_neumann_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, num_neighbors);

            // Print error if there's not enough neighbors
            TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
                    "ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed));
            // Put the values of alpha in the proper place in the global matrix
            for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
                local_index_type row = local_to_dof_map(_boundary_filtered_flags(i), field_one, k);
                for (local_index_type l=0; l<num_neighbors; l++) {
                    for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                        col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map(neighborhood->getNeighbor(_boundary_filtered_flags(i), l), field_two, n);
                        // Taking the dot product between curl(curl(velocity)) and normal vector
                        scalar_type collapsed_value = 0.0;
                        for (local_index_type m=0; m<fields[field_two]->nDim(); m++) {
                            // Obtained the normal direction
                            scalar_type normal_comp = _pressure_neumann_GMLS->getTangentBundle(i, 2, m);
                            collapsed_value -= b_i*normal_comp*_velocity_all_GMLS->getAlpha1TensorTo1Tensor(TargetOperation::CurlCurlOfVectorPointEvaluation, _boundary_filtered_flags(i), m /* output component*/, l, n /*input component*/);
                        }
                        val_data(l*fields[field_two]->nDim() + n) = collapsed_value;
                    }
                }
                {
                    this->_A->sumIntoLocalValues(row, num_neighbors*fields[field_two]->nDim(), val_data.data(), col_data.data());
                }
            }
        });
    } else if (field_one == pressure_field_id && field_two == pressure_field_id) {
        // Put values into pressure-pressure block

        // Put values from no-constraint GMLS into matrix
        int nlocal_internal = _internal_filtered_flags.extent(0);
        Kokkos::parallel_for(host_team_policy(nlocal_internal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level, Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
            const int i = teamMember.league_rank();

            host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
            host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

            const local_index_type num_neighbors = neighborhood->getNumNeighbors(_internal_filtered_flags(i));

            // Print error if there's not enough neighbors
            TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
                    "ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed));
            // Put the values of alpha in the proper place in the global matrix
            for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
                local_index_type row = local_to_dof_map(_internal_filtered_flags(i), field_one, k);
                for (local_index_type l=0; l<num_neighbors; l++) {
                    for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                        col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map(neighborhood->getNeighbor(_internal_filtered_flags(i), l), field_two, n);
                        if (n==k) { // same field, same component
                            val_data(l*fields[field_two]->nDim() + n) = _pressure_all_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, _internal_filtered_flags(i), l)*_pressure_all_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample,_internal_filtered_flags(i), l, false, 0, 0);
                            val_data(0) += _pressure_all_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, _internal_filtered_flags(i), l)*_pressure_all_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, _internal_filtered_flags(i), l, true, 0, 0);
                        } else {
                            val_data(l*fields[field_two]->nDim() + n) = 0.0;
                        }
                    }
                }
                {
                    this->_A->sumIntoLocalValues(row, num_neighbors*fields[field_two]->nDim(), val_data.data(), col_data.data());
                }
            }
        });
        Kokkos::fence();
        // Put values from Neumann GMLS into matrix
        int nlocal_boundary = _boundary_filtered_flags.extent(0);
        Kokkos::parallel_for(host_team_policy(nlocal_boundary, Kokkos::AUTO).set_scratch_size(host_scratch_team_level, Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
            const int i = teamMember.league_rank();

            host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
            host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

            const local_index_type num_neighbors = neighborhood->getNumNeighbors(_boundary_filtered_flags(i));

            // Print error if there's not enough neighbors
            TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
                    "ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed));
            // Put the values of alpha in the proper place in the global matrix
            for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
                local_index_type row = local_to_dof_map(_boundary_filtered_flags(i), field_one, k);
                for (local_index_type l=0; l<num_neighbors; l++) {
                    for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                        col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map(neighborhood->getNeighbor(_boundary_filtered_flags(i), l), field_two, n);
                        if (n==k) { // same field, same component
                            val_data(l*fields[field_two]->nDim() + n) = _pressure_neumann_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l)*_pressure_neumann_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample,i, l, false, 0, 0);
                            val_data(0) += _pressure_neumann_GMLS->getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l)*_pressure_neumann_GMLS->getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0, 0);
                        } else {
                            val_data(l*fields[field_two]->nDim() + n) = 0.0;
                        }
                    }
                }
                {
                    this->_A->sumIntoLocalValues(row, num_neighbors*fields[field_two]->nDim(), val_data.data(), col_data.data());
                }
            }
        });
    } else if (field_one == lm_pressure_field_id && field_two == pressure_field_id) {
        // row all DOFs for solution against Lagrange multiplier
        Teuchos::Array<local_index_type> col_data(nlocal*fields[field_two]->nDim());
        Teuchos::Array<scalar_type> val_data(nlocal*fields[field_two]->nDim());
        Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
        Teuchos::ArrayView<scalar_type> vals = Teuchos::ArrayView<scalar_type>(val_data);

        for (local_index_type l=0; l<nlocal; l++) {
            for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                cols[l*fields[field_two]->nDim() + n] = local_to_dof_map(l, field_two, n);
                vals[l*fields[field_two]->nDim() + n] = 1;
            }
        }

        auto comm = this->_particles->getCoordsConst()->getComm();
        if (comm->getRank()==0) {
            // local index 0 is the shared global id for the Lagrange multiplier
            this->_A->sumIntoLocalValues(0, cols, vals);
        } else {
            // local index 0 is the shared global id for the Lagrange multiplier
            this->_A->sumIntoLocalValues(nlocal, cols, vals);
        }
    } else if (field_one == pressure_field_id && field_two == lm_pressure_field_id) {
        // col all DOFs for solution against Lagrange multiplier
        auto comm = this->_particles->getCoordsConst()->getComm();
        for (local_index_type i=0; i<nlocal; i++) {
            for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
                local_index_type row = local_to_dof_map(i, field_one, k);

                Teuchos::Array<local_index_type> col_data(1);
                Teuchos::Array<scalar_type> val_data(1);
                Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
                Teuchos::ArrayView<scalar_type> vals = Teuchos::ArrayView<scalar_type>(val_data);

                if (comm->getRank() == 0) {
                    cols[0] = 0; // local index 0 is global shared index
                } else {
                    cols[0] = nlocal;
                }
                vals[0] = 1;
                this->_A->sumIntoLocalValues(row, cols, vals);
            }
        }
    } else if (field_one == lm_pressure_field_id && field_two == lm_pressure_field_id) {
        // identity on all DOFs for Lagrange multiplier
        const global_index_type nglobal = this->_coords->nGlobal();
        scalar_type eps_penalty = nglobal;

        auto comm = this->_particles->getCoordsConst()->getComm();
        for (local_index_type i=0; i<nlocal; i++) {
            for (local_index_type k=0; k<fields[field_two]->nDim(); k++) {
                local_index_type row = local_to_dof_map(i, field_one, k);

                Teuchos::Array<local_index_type> col_data(fields[field_two]->nDim());
                Teuchos::Array<scalar_type> val_data(fields[field_two]->nDim());
                Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
                Teuchos::ArrayView<scalar_type> vals = Teuchos::ArrayView<scalar_type>(val_data);

                for (local_index_type n=0; n<fields[field_two]->nDim(); n++) {
                    if (i==0 && comm->getRank()==0) {
                        cols[n] = local_to_dof_map(i, field_two, n);
                        vals[n] = eps_penalty;
                    } else {
                        cols[n] = local_to_dof_map(i, field_two, n);
                        vals[n] = 1.0;
                    }
                }
                {
                    this->_A->sumIntoLocalValues(row, cols, vals);
                }
            }
        }
    }
    Kokkos::fence();

    TEUCHOS_ASSERT(!this->_A.is_null());
    ComputeMatrixTime->stop();
}

const std::vector<InteractingFields> GMLS_StokesPhysics::gatherFieldInteractions() {
    std::vector<InteractingFields> field_interactions;

    field_interactions.push_back(InteractingFields(op_needing_interaction::physics,
        _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity")));
    field_interactions.push_back(InteractingFields(op_needing_interaction::physics,
        _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity"),
        _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure")));

    field_interactions.push_back(InteractingFields(op_needing_interaction::physics,
        _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure"),
        _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity")));
    field_interactions.push_back(InteractingFields(op_needing_interaction::physics,
        _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure")));

    field_interactions.push_back(InteractingFields(op_needing_interaction::physics,
        _particles->getFieldManagerConst()->getIDOfFieldFromName("lm_pressure")));
    field_interactions.push_back(InteractingFields(op_needing_interaction::physics,
        _particles->getFieldManagerConst()->getIDOfFieldFromName("lm_pressure"),
        _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure")));
    field_interactions.push_back(InteractingFields(op_needing_interaction::physics,
        _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure"),
        _particles->getFieldManagerConst()->getIDOfFieldFromName("lm_pressure")));

    return field_interactions;
}

}
