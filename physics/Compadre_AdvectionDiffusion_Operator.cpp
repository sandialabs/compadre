#include <Compadre_AdvectionDiffusion_Operator.hpp>

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
 Constructs the matrix operator.
 
 */
namespace Compadre {

typedef Compadre::CoordsT coords_type;
typedef Compadre::FieldT fields_type;
typedef Compadre::NeighborhoodT neighbors_type;
typedef Compadre::XyzVector xyz_type;


void AdvectionDiffusionPhysics::initialize() {

	Teuchos::RCP<Teuchos::Time> GenerateData = Teuchos::TimeMonitor::getNewCounter ("Generate Data");
    GenerateData->start();
    bool use_physical_coords = true; // can be set on the operator in the future

    // we don't know which particles are in each cell
    // cell neighbor search should someday be bigger than the particles neighborhoods (so that we are sure it includes all interacting particles)
    // get neighbors of cells (the neighbors are particles)
    local_index_type maxLeaf = _parameters->get<Teuchos::ParameterList>("neighborhood").get<int>("max leaf");

    // for now, since cells = particles, neighbor lists should be the same, later symmetric neighbor search should be enforced
    //_cell_particles_neighborhood = Teuchos::rcp_static_cast<neighbors_type>(Teuchos::rcp(
    //        new neighbors_type(_cells, _particles.getRawPtr(), false /*material coords*/, maxLeaf)));
    //_particle_cells_neighborhood = Teuchos::rcp_static_cast<neighbors_type>(Teuchos::rcp(
    //        new neighbors_type(_particles.getRawPtr(), _cells, false /*material coords*/, maxLeaf)));
    _cell_particles_neighborhood = Teuchos::rcp(_particles->getNeighborhood(), false /*owns memory*/);
    //_particles_particles_neighborhood = Teuchos::rcp_static_cast<neighbors_type>(Teuchos::rcp(
    //        new neighbors_type(_particles.getRawPtr(), _particles.getRawPtr(), false /*material coords*/, maxLeaf)));

    //auto neighbors_needed = GMLS::getNP(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 2);

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_particles->getCoordsConst()->getComm()->getRank()>0, "Only for serial.");
    //_particles_particles_neighborhood->constructAllNeighborLists(0, 
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
    //        true /*dry run for sizes*/,
    //        neighbors_needed,
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
    //        false);
    //_cell_particles_neighborhood->constructAllNeighborLists(0, 
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
    //        true /*dry run for sizes*/,
    //        neighbors_needed,
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
    //        false);
    //_particle_cells_neighborhood->constructAllNeighborLists(0, 
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
    //        true /*dry run for sizes*/,
    //        neighbors_needed,
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
    //        _parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
    //        false);

    //****************
    //
    //  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
    //
    //****************

    // generate the interpolation operator and call the coefficients needed (storing them)
    const coords_type* target_coords = this->_coords;
    const coords_type* source_coords = this->_coords;

    double max_window = _cell_particles_neighborhood->computeMaxHSupportSize(false /*local processor max*/);

    //_particles_particles_max_num_neighbors = 
    //    _particles_particles_neighborhood->computeMaxNumNeighbors(false /*local processor max*/);

    //_particle_cells_max_num_neighbors = 
    //    _particle_cells_neighborhood->computeMaxNumNeighbors(false /*local processor max*/);

    _cell_particles_max_num_neighbors = 
        _cell_particles_neighborhood->computeMaxNumNeighbors(false /*local processor max*/);

    Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), _cell_particles_max_num_neighbors+1);
    _kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);
    // fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        const int num_i_neighbors = _cell_particles_neighborhood->getNumNeighbors(i);
        for (int j=1; j<num_i_neighbors+1; ++j) {
            _kokkos_neighbor_lists_host(i,j) = _cell_particles_neighborhood->getNeighbor(i,j-1);
        }
        _kokkos_neighbor_lists_host(i,0) = num_i_neighbors;
    });

    Kokkos::View<double**> kokkos_augmented_source_coordinates("source_coordinates", source_coords->nLocal(true /* include halo in count */), source_coords->nDim());
    _kokkos_augmented_source_coordinates_host = Kokkos::create_mirror_view(kokkos_augmented_source_coordinates);
    // fill in the source coords, adding regular with halo coordiantes into a kokkos view
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int i) {
        xyz_type coordinate = source_coords->getLocalCoords(i, true /*include halo*/, use_physical_coords);
        _kokkos_augmented_source_coordinates_host(i,0) = coordinate.x;
        _kokkos_augmented_source_coordinates_host(i,1) = coordinate.y;
        _kokkos_augmented_source_coordinates_host(i,2) = coordinate.z;
    });

    Kokkos::View<double**> kokkos_target_coordinates("target_coordinates", target_coords->nLocal(), target_coords->nDim());
    _kokkos_target_coordinates_host = Kokkos::create_mirror_view(kokkos_target_coordinates);
    // fill in the target, adding regular coordiantes only into a kokkos view
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        xyz_type coordinate = target_coords->getLocalCoords(i, false /*include halo*/, use_physical_coords);
        _kokkos_target_coordinates_host(i,0) = coordinate.x;
        _kokkos_target_coordinates_host(i,1) = coordinate.y;
        _kokkos_target_coordinates_host(i,2) = coordinate.z;
    });

    auto epsilons = _cell_particles_neighborhood->getHSupportSizes()->getLocalView<const host_view_type>();
    Kokkos::View<double*> kokkos_epsilons("target_coordinates", target_coords->nLocal(), target_coords->nDim());
    _kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        _kokkos_epsilons_host(i) = epsilons(i,0);//0.75*max_window;//std::sqrt(max_window);//epsilons(i,0);
        //printf("epsilons %d: %f\n", i, epsilons(i,0));
    });

    // quantities contained on cells (mesh)
    auto quadrature_points = _cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_weights = _cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_type = _cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto unit_normals = _cells->getFieldManager()->getFieldByName("unit_normal")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto adjacent_elements = _cells->getFieldManager()->getFieldByName("adjacent_elements")->getMultiVectorPtr()->getLocalView<host_view_type>();




    // get all quadrature put together here as auxiliary evaluation sites
    _weights_ndim = quadrature_weights.extent(1);
    Kokkos::View<double**> kokkos_quadrature_coordinates("all quadrature coordinates", _cells->getCoordsConst()->nLocal()*_weights_ndim, 3);
    _kokkos_quadrature_coordinates_host = Kokkos::create_mirror_view(kokkos_quadrature_coordinates);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_cells->getCoordsConst()->nLocal()), KOKKOS_LAMBDA(const int i) {
        for (int j=0; j<_weights_ndim; ++j) {
            _kokkos_quadrature_coordinates_host(i*_weights_ndim + j, 0) = quadrature_points(i,2*j+0);
            _kokkos_quadrature_coordinates_host(i*_weights_ndim + j, 1) = quadrature_points(i,2*j+1);
        }
    });

    //// get cell neighbors of particles, and add indices to these for quadrature
    //// loop over particles, get cells in neighborhood and get their quadrature
    //Kokkos::View<int**> kokkos_quadrature_neighbor_lists("quadrature neighbor lists", target_coords->nLocal(), _weights_ndim*_particle_cells_max_num_neighbors+1);
    //_kokkos_quadrature_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_quadrature_neighbor_lists);
    //Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
    //    _kokkos_quadrature_neighbor_lists_host(i,0) = static_cast<int>(particle_cells_all_neighbors[i].size());
    //    for (int j=0; j<kokkos_quadrature_neighbor_lists(i,0); ++j) {
    //        auto cell_num = static_cast<int>(particle_cells_all_neighbors[i][j].first);
    //        for (int k=0; k<_weights_ndim; ++k) {
    //            _kokkos_quadrature_neighbor_lists_host(i, 1 + j*_weights_ndim + k) = cell_num*_weights_ndim + k;
    //        }
    //    }
    //});
    // get cell neighbors of particles, and add indices to these for quadrature
    // loop over particles, get cells in neighborhood and get their quadrature
    Kokkos::View<int**> kokkos_quadrature_neighbor_lists("quadrature neighbor lists", _cells->getCoordsConst()->nLocal(), _weights_ndim+1);
    _kokkos_quadrature_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_quadrature_neighbor_lists);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_cells->getCoordsConst()->nLocal()), KOKKOS_LAMBDA(const int i) {
        _kokkos_quadrature_neighbor_lists_host(i,0) = static_cast<int>(_weights_ndim);
        for (int j=0; j<kokkos_quadrature_neighbor_lists(i,0); ++j) {
            _kokkos_quadrature_neighbor_lists_host(i, 1+j) = i*_weights_ndim + j;
        }
    });

    //****************
    //
    // End of data copying
    //
    //****************

    // GMLS operator
    _gmls = Teuchos::rcp<GMLS>(new GMLS(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                 2 /* dimension */,
                 _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"), 
                 "STANDARD", "NO_CONSTRAINT"));
    _gmls->setProblemData(_kokkos_neighbor_lists_host,
                    _kokkos_augmented_source_coordinates_host,
                    _kokkos_target_coordinates_host,
                    _kokkos_epsilons_host);
    _gmls->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
    _gmls->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));

    _gmls->addTargets(TargetOperation::ScalarPointEvaluation);
    _gmls->setAdditionalEvaluationSitesData(_kokkos_quadrature_neighbor_lists_host, _kokkos_quadrature_coordinates_host);
    _gmls->generateAlphas();

    GenerateData->stop();

}

local_index_type AdvectionDiffusionPhysics::getMaxNumNeighbors() {
    return _cell_particles_max_num_neighbors;
}

Teuchos::RCP<crs_graph_type> AdvectionDiffusionPhysics::computeGraph(local_index_type field_one, local_index_type field_two) {
	if (field_two == -1) {
		field_two = field_one;
	}

	Teuchos::RCP<Teuchos::Time> ComputeGraphTime = Teuchos::TimeMonitor::getNewCounter ("Compute Graph Time");
	ComputeGraphTime->start();
	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A_graph.is_null(), "Tpetra CrsGraph for Physics not yet specified.");

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

	//#pragma omp parallel for
    for(local_index_type i = 0; i < nlocal; i++) {
        local_index_type num_neighbors = _cell_particles_neighborhood->getNumNeighbors(i);
        for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
            local_index_type row = local_to_dof_map(i, field_one, k);

            Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
            Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

            for (local_index_type l = 0; l < num_neighbors; l++) {
                for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                    col_data[l*fields[field_two]->nDim() + n] = local_to_dof_map(_cell_particles_neighborhood->getNeighbor(i,l), field_two, n);
                }
            }
            //#pragma omp critical
            {
                this->_A_graph->insertLocalIndices(row, cols);
            }
        }
    }
	ComputeGraphTime->stop();
    return this->_A_graph;
}


void AdvectionDiffusionPhysics::computeMatrix(local_index_type field_one, local_index_type field_two, scalar_type time) {

    bool use_physical_coords = true; // can be set on the operator in the future

    Teuchos::RCP<Teuchos::Time> ComputeMatrixTime = Teuchos::TimeMonitor::getNewCounter ("Compute Matrix Time");
    ComputeMatrixTime->start();

    TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A.is_null(), "Tpetra CrsMatrix for Physics not yet specified.");

    //Loop over all particles, convert to GMLS data types, solve problem, and insert into matrix:

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
    const host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();

    // quantities contained on cells (mesh)
    auto quadrature_points = _cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_weights = _cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_type = _cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto unit_normals = _cells->getFieldManager()->getFieldByName("unit_normal")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto adjacent_elements = _cells->getFieldManager()->getFieldByName("adjacent_elements")->getMultiVectorPtr()->getLocalView<host_view_type>();


    // loop over cells
    //Kokkos::View<int*, Kokkos::HostSpace> row_seen("has row had dirichlet addition", target_coords->nLocal());
    //host_vector_local_index_type row_seen("has row had dirichlet addition", nlocal);
    //Kokkos::deep_copy(row_seen,0);
    //host_vector_local_index_type col_data("col data", _cell_particles_max_num_neighbors);//particles_particles_max_num_neighbors*fields[field_two]->nDim());
    //host_vector_scalar_type val_data("val data", _cell_particles_max_num_neighbors);//particles_particles_max_num_neighbors*fields[field_two]->nDim());

    //auto penalty = (_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")+1)*_parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")*_particles_particles_neighborhood->getMinimumHSupportSize();
    //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")*_particles_particles_neighborhood->getMinimumHSupportSize();

    // each element must have the same number of edges
    auto num_edges = adjacent_elements.extent(1);
    auto num_interior_quadrature = 0;
    for (int q=0; q<_weights_ndim; ++q) {
        if (quadrature_type(0,q)!=1) { // some type of edge quadrature
            num_interior_quadrature = q;
            break;
        }
    }
    auto num_exterior_quadrature_per_edge = (_weights_ndim - num_interior_quadrature)/num_edges;
    //printf("num interior quadrature: %d\n", num_interior_quadrature);
    //printf("num exterior quadrature (per edge): %d\n", num_exterior_quadrature_per_edge);

    //double area = 0;
    //for (int i=0; i<_cells->getCoordsConst()->nLocal(); ++i) {
    //    for (int q=0; q<_weights_ndim; ++q) {
    //        if (quadrature_type(i,q)==1) { // interior
    //            area += quadrature_weights(i,q);
    //        }
    //    }
    //    // get all particle neighbors of cell i
    //    for (size_t j=0; j<cell_particles_all_neighbors[i].size(); ++j) {
    //        auto particle_j = cell_particles_all_neighbors[i][j].first;
    //        for (size_t k=0; k<cell_particles_all_neighbors[i].size(); ++k) {
    //            auto particle_k = cell_particles_all_neighbors[i][k].first;

    //            local_index_type row = local_to_dof_map(particle_j, field_one, 0 /* component 0*/);
    //            col_data(0) = local_to_dof_map(particle_k, field_two, 0 /* component */);
    //            local_index_type corresponding_particle_id = row;
    //            // loop over quadrature
    //            std::vector<double> edge_lengths(3);
    //            int current_edge_num = 0;
    //            for (int q=0; q<_weights_ndim; ++q) {
    //                if (quadrature_type(i,q)!=1) { // edge
    //                    int new_current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
    //                    if (new_current_edge_num!=current_edge_num) {
    //                        edge_lengths[new_current_edge_num] = quadrature_weights(i,q);
    //                        current_edge_num = new_current_edge_num;
    //                    } else {
    //                        edge_lengths[current_edge_num] += quadrature_weights(i,q);
    //                    }
    //                }
    //            }
    //            double contribution = 0;
    //            for (int q=0; q<_weights_ndim; ++q) {
    //                //printf("i: %d, j: %d, k: %d, q: %d, contribution: %f\n", i, j, k, q, contribution);
    //                if (quadrature_type(i,q)==1) { // interior

    //                    // mass matrix
    //                    contribution += quadrature_weights(i,q) * _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1) * _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1);
    //                    //printf("interior\n");
    //                } 
    //                //else if (quadrature_type(i,q)==2) { // edge on exterior
    //                //    //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_kokkos_epsilons_host(i);
    //                //    int current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
    //                //    auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/edge_lengths[current_edge_num];
    //                //    int adjacent_cell = adjacent_elements(i,current_edge_num);
	//                //    TEUCHOS_ASSERT(adjacent_cell < 0);
    //                //    // penalties for edges of mass
    //                //    double jumpvr = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1);
    //                //    double jumpur = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1);
    //                //    contribution += penalty * quadrature_weights(i,q) * jumpvr * jumpur;
    //                //    //printf("exterior edge\n");

    //                //} else if (quadrature_type(i,q)==0)  { // edge on interior
    //                //    //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_kokkos_epsilons_host(i);
    //                //    //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_kokkos_epsilons_host(i);
    //                //    //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty");///_kokkos_epsilons_host(i);
    //                //    int current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
    //                //    auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/edge_lengths[current_edge_num];
    //                //    int adjacent_cell = adjacent_elements(i,current_edge_num);
    //                //    // this cell numbering is only valid on serial (one) processor runs
	//                //    TEUCHOS_ASSERT(adjacent_cell >= 0);
    //                //    // penalties for edges of mass
    //                //    double jumpvr = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1)
    //                //        -_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell, j, q+1);
    //                //    double jumpur = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1)
    //                //        -_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell, k, q+1);
    //                //    //if (i<adjacent_cell) {
    //                //    //    contribution += penalty * quadrature_weights(i,q) * jumpvr * jumpur; // other half will be added by other cell
    //                //    //}
    //                //    contribution += penalty * 0.5 * quadrature_weights(i,q) * jumpvr * jumpur; // other half will be added by other cell
    //                //    //printf("jumpvr: %f, jumpur: %f, penalty: %f, qw: %f\n", jumpvr, jumpur, penalty, quadrature_weights(i,q));
    //                //    //printf("interior edge\n");
    //                //}
    //                //printf("i: %d, j: %d, k: %d, q: %d, contribution: %f\n", i, j, k, q, contribution);
	//                TEUCHOS_ASSERT(contribution==contribution);
    //            }

    //            val_data(0) = contribution;
    //            {
    //                if (row_seen(row)==1) { // already seen
    //                   // rhs_vals(row,0) += contribution;
    //                    this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
    //                } else {
    //                    //this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
    //                    this->_A->replaceLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
    //                    //rhs_vals(row,0) = contribution;
    //                    row_seen(row) = 1;
    //                }
    //                //this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
    //            }
    //        }
    //    }
    //}

    double area = 0; // (good, no rewriting)
    //Kokkos::View<double,Kokkos::MemoryTraits<Kokkos::Atomic> > area; // scalar
	int team_scratch_size = host_scratch_vector_scalar_type::shmem_size(1); // values
	team_scratch_size += host_scratch_vector_local_index_type::shmem_size(1); // local column indices
	const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently
	Kokkos::parallel_reduce(host_team_policy(nlocal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember, scalar_type& t_area) {
		const int i = teamMember.league_rank();

		host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), 1);
		host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), 1);

     //for (int i=0; i<_cells->getCoordsConst()->nLocal(); ++i) {

         ////if (i==3199) {
         //   printf("eps of %d: %.16f\n", i, _cell_particles_neighborhood->getHSupportSize(i));
         //   auto less_than = 0;
         //   for (int j=0; j<_cells->getCoordsConst()->nLocal(); ++j) {
         //       //printf("n: %d\n", _cell_particles_neighborhood->getNeighbor(i,j));
         //       auto this_i = _cells->getCoordsConst()->getLocalCoords(_cell_particles_neighborhood->getNeighbor(i,0));
         //       auto this_j = _cells->getCoordsConst()->getLocalCoords(j);
         //       auto diff = std::sqrt(pow(this_i[0]-this_j[0],2) + pow(this_i[1]-this_j[1],2) + pow(this_i[2]-this_j[2],2));
         //       if (diff <_cell_particles_neighborhood->getHSupportSize(i)) {
         //           //printf("diff: %.16f\n", std::sqrt(pow(this_i[0]-this_j[0],2) + pow(this_i[1]-this_j[1],2)));
         //           less_than++;
         //       }
         //   }
         //   if (less_than!=_cell_particles_neighborhood->getNumNeighbors(i)) {
         //       printf("THEY ARE DIFFERENT!\n");
         //   }
         //   //for (int j=0; j<_cell_particles_neighborhood->getNumNeighbors(i); ++j) {
         //   //    printf("n: %d\n", _cell_particles_neighborhood->getNeighbor(i,j));
         //   //    auto this_i = _cells->getCoordsConst()->getLocalCoords(_cell_particles_neighborhood->getNeighbor(i,0));
         //   //    auto this_j = _cells->getCoordsConst()->getLocalCoords(_cell_particles_neighborhood->getNeighbor(i,j));
         //   //    printf("diff: %.16f\n", std::sqrt(pow(this_i[0]-this_j[0],2) + pow(this_i[1]-this_j[1],2)));
         //   //}
         ////}

         for (int q=0; q<_weights_ndim; ++q) {
             if (quadrature_type(i,q)==1) { // interior
                 t_area += quadrature_weights(i,q);
             }
         }
         // get all particle neighbors of cell i
         //for (int particle_j=0; particle_j<_cells->getCoordsConst()->nLocal(); ++particle_j) {
         for (int j=0; j<_cell_particles_neighborhood->getNumNeighbors(i); ++j) {
             auto particle_j = _cell_particles_neighborhood->getNeighbor(i,j);
             //for (int particle_k=0; particle_k<_cells->getCoordsConst()->nLocal(); ++particle_k) {
             for (int k=0; k<_cell_particles_neighborhood->getNumNeighbors(i); ++k) {
                 auto particle_k = _cell_particles_neighborhood->getNeighbor(i,k);

                 //if (j==k) {
                 local_index_type row = local_to_dof_map(particle_j, field_one, 0 /* component 0*/);
                 col_data(0) = local_to_dof_map(particle_k, field_two, 0 /* component */);

                 //// i can see j, but can j see i?
                 //bool j_see_i = false;
                 //for (int l=0; l<_cell_particles_neighborhood->getNumNeighbors(particle_j); ++l) {
                 //    if (_cell_particles_neighborhood->getNeighbor(particle_j,l) == i) {
                 //        j_see_i = true;
                 //        break;
                 //    }
                 //}
                 //// i can see k, but can k see i?
                 //bool k_see_i = false;
                 //for (int l=0; l<_cell_particles_neighborhood->getNumNeighbors(particle_k); ++l) {
                 //    if (_cell_particles_neighborhood->getNeighbor(particle_k,l) == i) {
                 //        k_see_i = true;
                 //        break;
                 //    }
                 //}

                 //// j can see i, and k can see i, but can j see k?
                 //bool j_see_k = false;
                 //for (size_t l=0; l<_cell_particles_neighborhood->getNumNeighbors(particle_j); ++l) {
                 //    if (_cell_particles_neighborhood->getNeighbor(particle_j,l) == particle_k) {
                 //        j_see_k = true;
                 //        break;
                 //    }
                 //}

                 //local_index_type j_to_i = -1;
                 //for (int l=0; l<_cell_particles_neighborhood->getNumNeighbors(i); ++l) {
                 //    if (_cell_particles_neighborhood->getNeighbor(i,l) == particle_j) {
                 //        j_to_i = l;
                 //        break;
                 //    }
                 //}
                 //// i can see k, but can k see i?
                 //local_index_type k_to_i = -1;
                 //for (int l=0; l<_cell_particles_neighborhood->getNumNeighbors(i); ++l) {
                 //    if (_cell_particles_neighborhood->getNeighbor(i,l) == particle_k) {
                 //        k_to_i = l;
                 //        break;
                 //    }
                 //}
                 //compadre_assert_release((j_see_i && k_see_i) &&  "Someone can't see i.");


                 ////if (j_see_k && k_see_i && j_see_i) {
                 //if (j_to_i>-1 && k_to_i>-1) {
                     double contribution = 0;
                     for (int q=0; q<_weights_ndim; ++q) {
                         if (quadrature_type(i,q)==1) { // interior
                             // mass matrix
                             contribution += quadrature_weights(i,q) 
                                 //* _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_i, 0) 
                                 //* _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_i, 0);
                                 * _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1) 
                                 * _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1);
                         } 
	                     TEUCHOS_ASSERT(contribution==contribution);
                     }

                     val_data(0) = contribution;
                     this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
                 //}
                 //{
                 //    if (row_seen(row)==1) { // already seen
                 //        this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
                 //    } else {
                 //        this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
                 //        //this->_A->replaceLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
                 //        //row_seen(row) = 1;
                 //    }
                 //}
                 //}
             }
         }
     }, Kokkos::Sum<scalar_type>(area));
    //double area = 0; // (alternate assembly, also good)
    //for (int i=0; i<_cells->getCoordsConst()->nLocal(); ++i) {
    //    for (int q=0; q<_weights_ndim; ++q) {
    //        if (quadrature_type(i,q)==1) { // interior
    //            area += quadrature_weights(i,q);
    //        }
    //    }
    //    local_index_type row = local_to_dof_map(i, field_one, 0 /* component 0*/);
    //    // treat i as shape function, not cell
    //    // get all particle neighbors of particle i
    //    size_t num_particle_neighbors = _cell_particles_neighborhood->getNumNeighbors(i);
    //    for (int j=0; j<num_particle_neighbors; ++j) {
    //        auto particle_j = _cell_particles_neighborhood->getNeighbor(i,j);

    //        col_data(j) = local_to_dof_map(particle_j, field_two, 0 /* component */);
    //        double entry_i_j = 0;

    //        // particle i only has support on its neighbors (since particles == cells right now)
    //        for (int cell_k=0; cell_k<_cells->getCoordsConst()->nLocal(); ++cell_k) {
    //        //for (int k=0; k<_cell_particles_neighborhood->getNumNeighbors(i); ++k) {
    //            //auto cell_k = _cell_particles_neighborhood->getNeighbor(i,k);

    //            // what neighbor locally is particle j to cell k
    //            int j_to_k = -1;
    //            int i_to_k = -1;

    //            // does this cell see particle j? we know it sees particle i
    //            for (size_t l=0; l<_cell_particles_neighborhood->getNumNeighbors(cell_k); ++l) {
    //                if (_cell_particles_neighborhood->getNeighbor(cell_k,l) == particle_j) {
    //                    j_to_k = l;
    //                    break;
    //                }
    //            }
    //            for (size_t l=0; l<_cell_particles_neighborhood->getNumNeighbors(cell_k); ++l) {
    //                if (_cell_particles_neighborhood->getNeighbor(cell_k,l) == i) {
    //                    i_to_k = l;
    //                    break;
    //                }
    //            }
    //            if (i_to_k>=0 && j_to_k>=0) {
    //                for (int q=0; q<_weights_ndim; ++q) {
    //                    if (quadrature_type(cell_k,q)==1) { // interior
    //                        // mass matrix
    //                        entry_i_j += quadrature_weights(cell_k,q) 
    //                            * _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, cell_k, i_to_k, q+1) 
    //                            * _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, cell_k, j_to_k, q+1);
    //                    } 
	//                    TEUCHOS_ASSERT(entry_i_j==entry_i_j);
    //                }

    //            }
    //        }
    //        val_data(j) = entry_i_j;
    //    }
    //    this->_A->sumIntoLocalValues(row, num_particle_neighbors, val_data.data(), col_data.data());//, /*atomics*/false);
    //}
    //double area = 0;
    //for (int i=0; i<_cells->getCoordsConst()->nLocal(); ++i) {
    //    for (int q=0; q<_weights_ndim; ++q) {
    //        if (quadrature_type(i,q)==1) { // interior
    //            area += quadrature_weights(i,q);
    //        }
    //    }
    //    local_index_type row = local_to_dof_map(i, field_one, 0 /* component 0*/);
    //    // treat i as shape function, not cell
    //    // get all particle neighbors of particle i
    //    size_t num_particle_neighbors = cell_particles_all_neighbors[i].size();
    //    for (size_t j=0; j<num_particle_neighbors; ++j) {
    //        auto particle_j = cell_particles_all_neighbors[i][j].first;

    //        col_data(j) = local_to_dof_map(particle_j, field_two, 0 /* component */);
    //        double entry_i_j = 0;

    //        // particle i only has support on its neighbors (since particles == cells right now)
    //        for (size_t k=0; k<cell_particles_all_neighbors[i].size(); ++k) {
    //            auto cell_k = cell_particles_all_neighbors[i][k].first;

    //            // what neighbor locally is particle j to cell k
    //            int j_to_k = -1;
    //            int i_to_k = -1;

    //            // does this cell see particle j? we know it sees particle i
    //            for (size_t l=0; l<cell_particles_all_neighbors[cell_k].size(); ++l) {
    //                if (cell_particles_all_neighbors[cell_k][l].first == particle_j) {
    //                    j_to_k = l;
    //                    break;
    //                }
    //            }
    //            for (size_t l=0; l<cell_particles_all_neighbors[cell_k].size(); ++l) {
    //                if (cell_particles_all_neighbors[cell_k][l].first == i) {
    //                    i_to_k = l;
    //                    break;
    //                }
    //            }

    //            j_to_k = (j==0) ? i_to_k : -1;
    //            if (j_to_k>=0) {
    //                for (int q=0; q<_weights_ndim; ++q) {
    //                    if (quadrature_type(cell_k,q)==1) { // interior
    //                        // mass matrix
    //                        entry_i_j += quadrature_weights(cell_k,q) 
    //                            * _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, cell_k, i_to_k, q+1)
    //                            * 1;
    //                            //* _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, cell_k, j_to_k, q+1);
    //                    } 
	//                    TEUCHOS_ASSERT(entry_i_j==entry_i_j);
    //                }

    //            }
    //        }
    //        val_data(j) = entry_i_j;
    //    }
    //    this->_A->sumIntoLocalValues(row, num_particle_neighbors, val_data.data(), col_data.data());//, /*atomics*/false);
    //}
    printf("AREA: %.16f\n", area);

	TEUCHOS_ASSERT(!this->_A.is_null());
	ComputeMatrixTime->stop();

}

const std::vector<InteractingFields> AdvectionDiffusionPhysics::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	return field_interactions;
}
    
}
