#include <Compadre_AdvectionDiffusion_Operator.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_MultiJumpNeighborhood.hpp>
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
typedef Compadre::XyzVector xyz_type;


void AdvectionDiffusionPhysics::initialize() {

	Teuchos::RCP<Teuchos::Time> GenerateData = Teuchos::TimeMonitor::getNewCounter ("Generate Data");
    GenerateData->start();
    bool use_physical_coords = true; // can be set on the operator in the future

    // we don't know which particles are in each cell
    // cell neighbor search should someday be bigger than the particles neighborhoods (so that we are sure it includes all interacting particles)
    // get neighbors of cells (the neighbors are particles)
    local_index_type maxLeaf = _parameters->get<Teuchos::ParameterList>("neighborhood").get<int>("max leaf");
    _cell_particles_neighborhood = Teuchos::rcp(_particles->getNeighborhood(), false /*owns memory*/);

    // cell k can see particle i and cell k can see particle i
    // sparsity graph then requires particle i can see particle j since they will have a cell shared between them
    // since both i and j are shared by k, the radius for k's search doubled (and performed at i) will always find j
    // also, if it appears doubling is too costly, realize nothing less than double could be guaranteed to work
    // 
    // the alternative would be to perform a neighbor search from particles to cells, then again from cells to
    // particles (which would provide exactly what was needed)
    // in that case, for any neighbor j of i found, you can assume that there is some k approximately half the 
    // distance between i and j, which means that if doubling the search size is wasteful, it is barely so
 
    //
    // double-sized search
    //

    //auto max_h = _cell_particles_neighborhood->computeMaxHSupportSize(false /* local processor max is fine, since they are all the same */);
    //auto neighbors_needed = GMLS::getNP(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 2);
    //_particles_double_hop_neighborhood = Teuchos::rcp_static_cast<neighborhood_type>(Teuchos::rcp(
    //        new neighborhood_type(_cells, _particles.getRawPtr(), false /*material coords*/, maxLeaf)));
    //_particles_double_hop_neighborhood->constructAllNeighborLists(1e+16,
    //    "radius",
    //    true /*dry run for sizes*/,
    //    neighbors_needed+1,
    //    0.0, /* cutoff multiplier */
    //    2*max_h, /* search size */
    //    false, /* uniform radii is true, but will be enforce automatically because all search sizes are the same globally */
    //    _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));


    //
    // neighbor of neighbor search
    // 

    _particles_double_hop_neighborhood = Teuchos::rcp(new MultiJumpNeighborhood(_cell_particles_neighborhood.getRawPtr()));
    auto as_multi_jump = Teuchos::rcp_static_cast<MultiJumpNeighborhood>(_particles_double_hop_neighborhood);
    as_multi_jump->constructNeighborOfNeighborLists(_cells->getCoordsConst()->getHaloSize());


    //TEUCHOS_TEST_FOR_EXCEPT_MSG(_particles->getCoordsConst()->getComm()->getRank()>0, "Only for serial.");

    //****************
    //
    //  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
    //
    //****************

    // generate the interpolation operator and call the coefficients needed (storing them)
    const coords_type* target_coords = this->_coords;
    const coords_type* source_coords = this->_coords;

    double max_window = _cell_particles_neighborhood->computeMaxHSupportSize(false /*local processor max*/);

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
    return _particles_double_hop_neighborhood->computeMaxNumNeighbors(false /*local processor maximum*/);
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

    for(local_index_type i = 0; i < nlocal; i++) {
        local_index_type num_neighbors = _particles_double_hop_neighborhood->getNumNeighbors(i);
        for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
            local_index_type row = local_to_dof_map(i, field_one, k);

            Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
            Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

            for (local_index_type l = 0; l < num_neighbors; l++) {
                for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                    col_data[l*fields[field_two]->nDim() + n] = local_to_dof_map(_particles_double_hop_neighborhood->getNeighbor(i,l), field_two, n);
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
	team_scratch_size += host_scratch_vector_local_index_type::shmem_size(_cell_particles_neighborhood->computeMaxNumNeighbors(false)); // local column indices
	//const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently
	//Kokkos::parallel_reduce(host_team_policy(nlocal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember, scalar_type& t_area) {
	//	const int i = teamMember.league_rank();

	//	host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), 1);
	//	host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), 1);
	//	host_scratch_vector_local_index_type more_than(teamMember.team_scratch(host_scratch_team_level), _cell_particles_neighborhood->getNumNeighbors(i));

    // loop over cells
    host_vector_local_index_type col_data("col data", _cell_particles_max_num_neighbors);//particles_particles_max_num_neighbors*fields[field_two]->nDim());
    host_vector_local_index_type more_than("more than", _cell_particles_max_num_neighbors);//particles_particles_max_num_neighbors*fields[field_two]->nDim());
    host_vector_scalar_type val_data("val data", _cell_particles_max_num_neighbors);//particles_particles_max_num_neighbors*fields[field_two]->nDim());
    double t_area = 0; 
    for (int i=0; i<_cells->getCoordsConst()->nLocal(); ++i) {
        //auto window_size = _cell_particles_neighborhood->getHSupportSize(i);
        ////auto less_than = 0;
        //for (int j=0; j<_cell_particles_neighborhood->getNumNeighbors(i); ++j) {
        //    auto this_i = _cells->getCoordsConst()->getLocalCoords(_cell_particles_neighborhood->getNeighbor(i,0));
        //    auto this_j = _cells->getCoordsConst()->getLocalCoords(_cell_particles_neighborhood->getNeighbor(i,j));
        //    auto diff = std::sqrt(pow(this_i[0]-this_j[0],2) + pow(this_i[1]-this_j[1],2) + pow(this_i[2]-this_j[2],2));
        //    if (diff > window_size) {
        //        //printf("diff: %.16f\n", std::sqrt(pow(this_i[0]-this_j[0],2) + pow(this_i[1]-this_j[1],2)));
        //        more_than(j) = 1;
        //    } else {
        //        more_than(j) = 0;
        //    }
        //}

         for (int q=0; q<_weights_ndim; ++q) {
             if (quadrature_type(i,q)==1) { // interior
                 t_area += quadrature_weights(i,q);
             }
         }
         // get all particle neighbors of cell i
         for (int j=0; j<_cell_particles_neighborhood->getNumNeighbors(i); ++j) {
             auto particle_j = _cell_particles_neighborhood->getNeighbor(i,j);
             for (int k=0; k<_cell_particles_neighborhood->getNumNeighbors(i); ++k) {
                 auto particle_k = _cell_particles_neighborhood->getNeighbor(i,k);

                 //if (j==k) {
                 local_index_type row = local_to_dof_map(particle_j, field_one, 0 /* component 0*/);
                 col_data(0) = local_to_dof_map(particle_k, field_two, 0 /* component */);



                 double contribution = 0;
                 if (_parameters->get<Teuchos::ParameterList>("physics").get<bool>("l2 projection only")) {
                    for (int q=0; q<_weights_ndim; ++q) {
                        if (quadrature_type(i,q)==1) { // interior
                            contribution += quadrature_weights(i,q) 
                                * _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1) 
                                * _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1);
                        } 
	                    TEUCHOS_ASSERT(contribution==contribution);
                    }
                 }

                 val_data(0) = contribution;
                 this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
             }
         }
     }
     area = t_area;
     //}, Kokkos::Sum<scalar_type>(area));


    //double area = 0; // (alternate assembly, also good)
    ////Kokkos::View<double,Kokkos::MemoryTraits<Kokkos::Atomic> > area; // scalar
	//int team_scratch_size = host_scratch_vector_scalar_type::shmem_size(_cell_particles_max_num_neighbors); // values
	//team_scratch_size += host_scratch_vector_local_index_type::shmem_size(_cell_particles_max_num_neighbors); // local column indices
	//const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently
	//Kokkos::parallel_reduce(host_team_policy(nlocal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember, scalar_type& t_area) {
	//	const int i = teamMember.league_rank();

	//	host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), _cell_particles_max_num_neighbors);
	//	host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), _cell_particles_max_num_neighbors);

    //    for (int q=0; q<_weights_ndim; ++q) {
    //        if (quadrature_type(i,q)==1) { // interior
    //            t_area += quadrature_weights(i,q);
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
    //}, Kokkos::Sum<scalar_type>(area));
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
