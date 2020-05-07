#include <Compadre_ReactionDiffusion_Operator.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_MultiJumpNeighborhood.hpp>
#include <Compadre_XyzVector.hpp>

#include <Compadre_GMLS.hpp>

#ifdef TRILINOS_DISCRETIZATION
  #include <Shards_CellTopology.hpp>
  #include <Intrepid_DefaultCubatureFactory.hpp>
  #include <Intrepid_CellTools.hpp>
  #include <Intrepid_FunctionSpaceTools.hpp>
  #include <Intrepid_HGRAD_TRI_C2_FEM.hpp>
#endif

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


void ReactionDiffusionPhysics::initialize() {

#ifndef TRILINOS_DISCRETIZATION
    compadre_assert_release(false && "Trilinos packages Shards and Intrepid required to run this example.");
#else

    Teuchos::RCP<Teuchos::Time> GenerateData = Teuchos::TimeMonitor::getNewCounter ("Generate Data");
    GenerateData->start();
    bool use_physical_coords = true; // can be set on the operator in the future
 
    auto num_cells_local = _cells->getCoordsConst()->nLocal();

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
 
    _ndim_requested = _parameters->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions");


    //
    // double and triple hop sized search
    //
    // it is a safe upper bound to use twice the previous search size as the maximum new radii
    _cell_particles_max_h = _cell_particles_neighborhood->computeMaxHSupportSize(true /* global processor max */);
    scalar_type triple_radius = 3.0 * _cell_particles_max_h;
    scalar_type max_search_size = _cells->getCoordsConst()->getHaloSize();
    // check that max_halo_size is not violated by distance of the double jump
    const local_index_type comm_size = _cell_particles_neighborhood->getSourceCoordinates()->getComm()->getSize();
    if (comm_size > 1) {
        TEUCHOS_TEST_FOR_EXCEPT_MSG((max_search_size < triple_radius), "Neighbor of neighbor search results in a search radius exceeding the halo size.");
    }

    auto neighbors_needed = GMLS::getNP(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), _ndim_requested);

    _particles_triple_hop_neighborhood = Teuchos::rcp_static_cast<neighborhood_type>(Teuchos::rcp(
            new neighborhood_type(_cells, _particles.getRawPtr(), false /*material coords*/, maxLeaf)));
    _particles_triple_hop_neighborhood->constructAllNeighborLists(max_search_size,
        "radius",
        true /*dry run for sizes*/,
        neighbors_needed+1,
        0.0, /* cutoff multiplier */
        triple_radius, /* search size */
        false, /* uniform radii is true, but will be enforce automatically because all search sizes are the same globally */
        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));
     Kokkos::fence();

    //****************
    //
    //  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
    //
    //****************

    // generate the interpolation operator and call the coefficients needed (storing them)
    const coords_type* target_coords = this->_coords;
    const coords_type* source_coords = this->_coords;

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
        if (_ndim_requested>2) _kokkos_augmented_source_coordinates_host(i,2) = coordinate.z;
    });

    Kokkos::View<double**> kokkos_target_coordinates("target_coordinates", target_coords->nLocal(), target_coords->nDim());
    _kokkos_target_coordinates_host = Kokkos::create_mirror_view(kokkos_target_coordinates);
    // fill in the target, adding regular coordiantes only into a kokkos view
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        xyz_type coordinate = target_coords->getLocalCoords(i, false /*include halo*/, use_physical_coords);
        _kokkos_target_coordinates_host(i,0) = coordinate.x;
        _kokkos_target_coordinates_host(i,1) = coordinate.y;
        if (_ndim_requested>2) _kokkos_target_coordinates_host(i,2) = coordinate.z;
    });

    auto epsilons = _cell_particles_neighborhood->getHSupportSizes()->getLocalView<const host_view_type>();
    Kokkos::View<double*> kokkos_epsilons("epsilons", target_coords->nLocal());
    _kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        _kokkos_epsilons_host(i) = epsilons(i,0);
    });















    // element that mesh consists of
    shards::CellTopology element = (_ndim_requested==2) ? shards::getCellTopologyData< shards::Triangle<> >() : shards::getCellTopologyData< shards::Tetrahedron<> >();

    int element_dim = element.getDimension();

    Intrepid::DefaultCubatureFactory<double> cubature_factory;
    int cubature_degree = 2*_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder");

    int num_element_nodes = element.getNodeCount();
    int num_element_sides = element.getSideCount();
    Teuchos::RCP<Intrepid::Cubature<double> > element_cubature = cubature_factory.create(element, cubature_degree);
    int num_element_cub_points = element_cubature->getNumPoints();

    // quadrature sites and weights on sides
    shards::CellTopology side = (_ndim_requested==2) ? shards::getCellTopologyData< shards::Line<> >() : shards::getCellTopologyData< shards::Triangle<> >();
    int num_side_nodes = side.getNodeCount();
    Teuchos::RCP<Intrepid::Cubature<double> > side_cubature = cubature_factory.create(side, cubature_degree);
    int num_side_cub_points = side_cubature->getNumPoints();

    _weights_ndim = num_element_cub_points + num_element_sides*(num_side_cub_points);

    // quantities contained on cells (mesh)
    _cells->getFieldManager()->createField(element_dim*_weights_ndim, "quadrature_points");
    _cells->getFieldManager()->createField(_weights_ndim, "quadrature_weights");
    _cells->getFieldManager()->createField(_weights_ndim, "interior");
    _cells->getFieldManager()->createField(element_dim*_weights_ndim, "unit_normal");
    auto quadrature_points = _cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_weights = _cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_type = _cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto unit_normals = _cells->getFieldManager()->getFieldByName("unit_normal")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto cell_vertices = _cells->getFieldManager()->getFieldByName("vertex_points")->getMultiVectorPtr()->getLocalView<host_view_type>();
    
    // this quantity refers to global elements, so it's local number can be found by getting neighbor with same global number
    auto adjacent_elements = _cells->getFieldManager()->getFieldByName("adjacent_elements")->getMultiVectorPtr()->getLocalView<host_view_type>();

    Intrepid::FieldContainer<double> element_cub_points(num_element_cub_points, element_dim);
    Intrepid::FieldContainer<double> element_cub_weights(num_element_cub_points);
    element_cubature->getCubature(element_cub_points, element_cub_weights);

    Intrepid::FieldContainer<double> element_nodes(num_cells_local, num_element_nodes, element_dim);
    //Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_cells->getCoordsConst()->nLocal()), KOKKOS_LAMBDA(const int i) {
    for (int i=0; i<num_cells_local; ++i) {
        for (int j=0; j<num_element_nodes; ++j) {
            for (int k=0; k<element_dim; ++k) {
                element_nodes(i, j, k) = cell_vertices(i, _ndim_requested*j + k);
            }
        }
    }//);
    Intrepid::FieldContainer<double> element_jacobian(num_cells_local, num_element_cub_points, element_dim, element_dim);
    Intrepid::FieldContainer<double> element_jacobian_det(num_cells_local, num_element_cub_points);
    Intrepid::FieldContainer<double> physical_element_cub_points(num_cells_local, num_element_cub_points, element_dim);
    Intrepid::FieldContainer<double> physical_element_cub_weights(num_cells_local, num_element_cub_points);

    Intrepid::CellTools<double>::setJacobian(element_jacobian, element_cub_points, element_nodes, element);
    Intrepid::CellTools<double>::setJacobianDet(element_jacobian_det, element_jacobian);
    Intrepid::FunctionSpaceTools::computeCellMeasure<double>(physical_element_cub_weights,
                                                             element_jacobian_det,
                                                             element_cub_weights);
    Intrepid::CellTools<scalar_type>::mapToPhysicalFrame(physical_element_cub_points,
                                                         element_cub_points,
                                                         element_nodes,
                                                         element,
                                                         -1);


    int side_dim = side.getDimension();
    Intrepid::FieldContainer<double> side_cub_points_nm1d(num_side_cub_points, side_dim);
    Intrepid::FieldContainer<double> side_cub_weights(num_side_cub_points);
    side_cubature->getCubature(side_cub_points_nm1d, side_cub_weights);

    std::vector<Intrepid::FieldContainer<double> > side_cub_points_nd(num_element_sides, Intrepid::FieldContainer<double>(num_side_cub_points, element_dim));
    for (int i=0; i<num_element_sides; ++i) {
        Intrepid::CellTools<double>::mapToReferenceSubcell(side_cub_points_nd[i],
                                                           side_cub_points_nm1d,
                                                           side_dim,
                                                           i,
                                                           element);
    }

    std::vector<Intrepid::FieldContainer<double> > side_jacobian(num_element_sides, Intrepid::FieldContainer<double>(num_cells_local, num_side_cub_points, element_dim, element_dim));
    std::vector<Intrepid::FieldContainer<double> > physical_side_cub_weights(num_element_sides, Intrepid::FieldContainer<double>(num_cells_local, num_side_cub_points));
    std::vector<Intrepid::FieldContainer<double> > physical_side_cub_points(num_element_sides, Intrepid::FieldContainer<double>(num_cells_local, num_side_cub_points, element_dim));
    std::vector<Intrepid::FieldContainer<double> > side_normals(num_element_sides, Intrepid::FieldContainer<double>(num_cells_local, num_side_cub_points, element_dim));
    Kokkos::View<double***, Kokkos::HostSpace> side_unit_normals("side unit normals", num_cells_local, num_element_sides, element_dim);
    std::vector<Intrepid::FieldContainer<double> > side_unit_normals_int(num_element_sides, Intrepid::FieldContainer<double>(num_cells_local, num_side_cub_points, element_dim));

    for (int i=0; i<num_element_sides; ++i) {
        Intrepid::CellTools<double>::setJacobian(side_jacobian[i], side_cub_points_nd[i], element_nodes, element);
        if (_ndim_requested==2) {
            Intrepid::FunctionSpaceTools::computeEdgeMeasure<double>(physical_side_cub_weights[i],
                                                                     side_jacobian[i],
                                                                     side_cub_weights,
                                                                     i,
                                                                     element);
        } else {
            Intrepid::FunctionSpaceTools::computeFaceMeasure<double>(physical_side_cub_weights[i],
                                                                     side_jacobian[i],
                                                                     side_cub_weights,
                                                                     i,
                                                                     element);
        }
        Intrepid::CellTools<scalar_type>::mapToPhysicalFrame(physical_side_cub_points[i],
                                                             side_cub_points_nd[i],
                                                             element_nodes,
                                                             element,
                                                             -1);
        Intrepid::CellTools<scalar_type>::getPhysicalSideNormals(side_normals[i],
                                                                 side_jacobian[i],
                                                                 i,
                                                                 element);
    }
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_cells_local), KOKKOS_LAMBDA(const int i) {
        // calculated for only one cubature point per side because it is done on a simplex
        // if ever done on non-simplicial mesh, would require normal at each cubature point.
        for (int j=0; j<num_element_sides; ++j) {
            double vector_norm = 0;
            for (int k=0; k<element_dim; ++k) {
                vector_norm += side_normals[j](i, 0, k)*side_normals[j](i, 0, k);
            }
            vector_norm = std::sqrt(vector_norm);
            for (int k=0; k<element_dim; ++k) {
                side_unit_normals(i, j, k) = side_normals[j](i, 0, k)/vector_norm;
            }
        }
    });


    

    if (_use_vms) {
    
        // Compute Jacobian Inverse (needed to compute gradients wrt physical coords)
        Intrepid::FieldContainer<double> element_jacobian_inv(num_cells_local, num_element_cub_points, element_dim, element_dim); //allocate
        Intrepid::CellTools<double>::setJacobianInv(element_jacobian_inv, element_jacobian); //compute
        
        // Generate Basis functions for fine scales
        // calling entire T6 basis, (we only need a subset of it)
        Intrepid::Basis_HGRAD_TRI_C2_FEM<double, Intrepid::FieldContainer<double> > basis_T6; // Define basis
        int num_basis_functions = basis_T6.getCardinality(); 
        
        // MA NOTE: Operations that follow apply to entire T6 basis, for Tau, we only want basis functions associated with edges, with DOF ordinals (3,4,5)
        // For now, it is easier (more efficient?) to use Intrepid to operate on the entire basis
        // We can pull/call only the entries associated with DOF ordinals(3,4,5) (assoc. w/ edges) at the end
        
        // GRADS of basis functions wrt REF coords (in one ref cell)
        Intrepid::FieldContainer<double> basis_T6_grad(num_basis_functions, num_element_cub_points, element_dim); //allocate 
        basis_T6.getValues(basis_T6_grad, element_cub_points, Intrepid::OPERATOR_GRAD); //compute
   
        // GRADS of basis functions wrt PHYSICAL coords (for all cells)
        Intrepid::FieldContainer<double> physical_basis_T6_grad(num_cells_local, num_basis_functions, num_element_cub_points, element_dim); //allocate 
        Intrepid::FunctionSpaceTools::HGRADtransformGRAD<double>(physical_basis_T6_grad, element_jacobian_inv, basis_T6_grad); //compute 
        
        // Multiply PHYS GRADS with CUB WEIGHTS
        Intrepid::FieldContainer<double> physical_basis_T6_grad_weighted(num_cells_local, num_basis_functions, num_element_cub_points, element_dim); //allocate 
        Intrepid::FunctionSpaceTools::multiplyMeasure<double>(physical_basis_T6_grad_weighted, physical_element_cub_weights, physical_basis_T6_grad); //compute
        
        // Modified bubbles (quadratic edge bubbles raised to some power)
        // Make the power an integer, so that the modified bubble order is larger than the order of the underlying polynomial approximation in GMLS
        double bub_pow = 2.0; //exponent
        Intrepid::FieldContainer<double> basis_T6_values(num_basis_functions, num_element_cub_points); //allocate values
        basis_T6.getValues(basis_T6_values, element_cub_points, Intrepid::OPERATOR_VALUE); //compute
        int size_basis_T6_values = basis_T6_values.size();
        Intrepid::FieldContainer<double> basis_T6_values_pow_m1(num_basis_functions, num_element_cub_points); //allocate mod. vals.
        //parallelize? (maybe compiler vectorizes loop already)
        for (int k = 0; k < size_basis_T6_values; ++k) {
            basis_T6_values_pow_m1[k] = std::pow(basis_T6_values[k], bub_pow - 1.0); //raise to bub_pow-1
        }
        
        // GRAD of modified bubble (using chain rule): product nb^(n-1)*grad(b)
        // Did not find a "field-field" multiply capability in Intrepid FunctionSpaceTools. Use a loop. 
        Intrepid::FieldContainer<double> physical_basis_T6_grad_pow(num_cells_local, num_basis_functions, num_element_cub_points, element_dim); //allocate 
        Intrepid::FieldContainer<double> physical_basis_T6_grad_pow_weighted(num_cells_local, num_basis_functions, num_element_cub_points, element_dim); //allocate
        //parallelize
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_cells_local), [&](const int k1) {
            for (int k2 = 0; k2 < num_basis_functions; ++k2){
                for (int k3 = 0; k3 < num_element_cub_points; ++k3) {
                    for (int k4 = 0; k4 < element_dim; ++k4) {
                        physical_basis_T6_grad_pow(k1, k2, k3, k4) = bub_pow * basis_T6_values_pow_m1(k2, k3) * physical_basis_T6_grad(k1, k2, k3, k4); //multiply 
                        physical_basis_T6_grad_pow_weighted(k1, k2, k3, k4) = bub_pow * basis_T6_values_pow_m1(k2, k3) * physical_basis_T6_grad_weighted(k1, k2, k3, k4); //multiply
                    }
                }
            }
        });

        // 'Standard' Stiffness Matrices (may be unnecessary, and may be deprecated some day) 
        Intrepid::FieldContainer<double> Stiff_Matrices_pow(num_cells_local, num_basis_functions, num_basis_functions);
        Intrepid::FunctionSpaceTools::integrate<double> (Stiff_Matrices_pow, physical_basis_T6_grad_pow_weighted, physical_basis_T6_grad_pow, Intrepid::COMP_CPP);
        
        // MA NOTE: Still need to compute terms for off-diagonal entries for tau: integral (db/dxi * db/dxj)
        // In order to compute this integrals with Intrepid, need to restructure the arrays with GRADS of basis functions, as follows
 
        // Use vector of FieldContainers, each vector entry corresponds to a finse-scale basis function. (sep: separated)
        std::vector<Intrepid::FieldContainer<double> > sep_physical_basis_T6_grad_pow(num_basis_functions, 
                Intrepid::FieldContainer<double>(num_cells_local, element_dim, num_element_cub_points));
        
        std::vector<Intrepid::FieldContainer<double> > sep_physical_basis_T6_grad_pow_weighted(num_basis_functions, 
                Intrepid::FieldContainer<double>(num_cells_local, element_dim, num_element_cub_points) );
        
        // Restructure arrays
        // some day, loop over only basis functions (k2) associated to edges
        // parallelize (be careful with index reordering)
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_cells_local), [&](const int k1) {
            for (int k2 = 0; k2 < num_basis_functions; ++k2){
                for (int k3 = 0; k3 < num_element_cub_points; ++k3) {
                    for (int k4 = 0; k4 < element_dim; ++k4) {
                        sep_physical_basis_T6_grad_pow[k2](k1, k4, k3) = physical_basis_T6_grad_pow(k1, k2, k3, k4);
                        sep_physical_basis_T6_grad_pow_weighted[k2](k1, k4, k3) = physical_basis_T6_grad_pow_weighted(k1, k2, k3, k4);
                    }
                }
            }
        });

        // Compute ndim x ndim matrices: integral (db/dxi * db/dxj)
        std::vector<Intrepid::FieldContainer<double> > sep_grad_grad_mat_integrals(num_basis_functions,
                Intrepid::FieldContainer<double>(num_cells_local, element_dim, element_dim) ); //allocate
        
        // could parallelize loop
        for (int k = 0; k < num_basis_functions; ++k) {
            Intrepid::FunctionSpaceTools::operatorIntegral<double>(sep_grad_grad_mat_integrals[k], sep_physical_basis_T6_grad_pow[k], 
                                                                   sep_physical_basis_T6_grad_pow_weighted[k], Intrepid::COMP_CPP); //integrate
        }

        // VALUES at EDGE CUB PTS
        std::vector<Intrepid::FieldContainer<double> > basis_T6_edge_values(num_element_sides, Intrepid::FieldContainer<double>(num_basis_functions, num_side_cub_points) ); //allocate 
        //parallelize? (small loop, maybe compiler already vectorices)
        for (int i=0; i<num_element_sides; ++i) {
            basis_T6.getValues(basis_T6_edge_values[i], side_cub_points_nd[i], Intrepid::OPERATOR_VALUE);    
        }
        
        // Modify bubble (raise to power)
        std::vector<Intrepid::FieldContainer<double> > basis_T6_edge_values_pow(num_element_sides, Intrepid::FieldContainer<double>(num_basis_functions, num_side_cub_points) ); //allocate
        int size_basis_T6_edge_values_1 = num_basis_functions * num_side_cub_points;
        //parallelize (maybe compiler already unrolls and vectorizes)
        for (int i=0; i<num_element_sides; ++i) {
            for (int k=0; k<size_basis_T6_edge_values_1; ++k) {
                basis_T6_edge_values_pow[i][k] = std::pow(basis_T6_edge_values[i][k], bub_pow); //exponentiate: (we start with quadratic bubble, then raised to a power to make order higher than underlying GMLS)
            }
        }
        
        //Multiply CUB WEIGHTS
        //side note: FunctionSpaceTools::multiplyMeasure is a wrappper for FunctionSpaceTools::scalarMultiplyDataField
        //target field container can be indexed as (C,F,P) or just as (F,P) (the latter applies here) 
        std::vector<Intrepid::FieldContainer<double> > basis_T6_edge_values_pow_weighted(num_element_sides,
            Intrepid::FieldContainer<double>(num_cells_local, num_basis_functions, num_side_cub_points) ); //allocate 
        for (int i=0; i<num_element_sides; ++i) {
            Intrepid::FunctionSpaceTools::multiplyMeasure<double>(basis_T6_edge_values_pow_weighted[i], physical_side_cub_weights[i], basis_T6_edge_values_pow[i]); //multiply
        }
        
        //INTEGRATE
        //To integrate just the value of the bubble basis, need to create a FieldContainer of only "1".
        //couldn't integrate directly with the inputs above because both array inputs to "integrate" need to have dimension num_cells_local in their rank 0
        std::vector<Intrepid::FieldContainer<double> > ones_for_edge_integration(num_element_sides, Intrepid::FieldContainer<double>(num_cells_local, num_side_cub_points) ); //allocate 
        for (int i=0; i<num_element_sides; ++i) {
            ones_for_edge_integration[i].initialize(1.0); //set ones
        }
         
        std::vector<Intrepid::FieldContainer<double> > basis_T6_edge_values_pow_integrated(num_element_sides,
                    Intrepid::FieldContainer<double>(num_cells_local, num_basis_functions) ); //allocate 
        for (int i=0; i<num_element_sides; ++i) {
            Intrepid::FunctionSpaceTools::functionalIntegral<double>(basis_T6_edge_values_pow_integrated[i],
                                              ones_for_edge_integration[i], basis_T6_edge_values_pow_weighted[i], Intrepid::COMP_CPP); //integrate
        }

        //Edge Lengths (Measures)
        std::vector<Intrepid::FieldContainer<double> > element_edge_lengths(num_element_sides, Intrepid::FieldContainer<double>(num_cells_local) ); //allocate
        
        // use FunctionSpaceTools:dataIntegral
        for (int i=0; i<num_element_sides; ++i) {
            Intrepid::FunctionSpaceTools::dataIntegral<double>(element_edge_lengths[i], ones_for_edge_integration[i], physical_side_cub_weights[i], Intrepid::COMP_CPP); //integrate
        }

        auto num_edges = _ndim_requested + 1;
        _tau = Kokkos::View<double****, Kokkos::HostSpace>("tau", num_cells_local, num_edges, _ndim_requested, _ndim_requested);

        //Forming matrix Amat (for scalar problem, it is a scalar)
        Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> Amat(_ndim_requested, _ndim_requested);
        Teuchos :: SerialDenseSolver <local_index_type, scalar_type> Amat_solver;
        
        //could parallelize outer two loops
        for (int i = 0; i < num_element_sides; ++i) {
        
            int edge_ordinal = i + 3; // this selects the "edge bubble" function associated with the corresponding edge
                
            for (int k = 0; k < num_cells_local; ++k) {
                
                Amat.putScalar(0.0);

                //First, compute auxiliary quantity: inner product of bubble gradient with itself
                double grad_bub_2;
                grad_bub_2 = 0.0;
                for (int j1 = 0; j1 < _ndim_requested; ++j1){
                    grad_bub_2 += sep_grad_grad_mat_integrals[edge_ordinal](k, j1, j1);
                }
                for (int j1 = 0; j1 < _ndim_requested; ++j1){
                    for (int j2 = 0; j2 < _ndim_requested; ++j2) {
                        //THIS IS TEMP (actual Amat involves material elastic parameters shear and lambda
                        // this is A_ij = db/dx_i * db/dx_j
                        // Amat(j1, j2) = sep_grad_grad_mat_integrals[edge_ordinal](k, j1, j2);
                        Amat(j1, j2) = ( _shear + _lambda / 2.0 ) * sep_grad_grad_mat_integrals[edge_ordinal](k, j1, j2);
                    }
                    Amat(j1, j1) += _shear * grad_bub_2;
                }

                Amat_solver.setMatrix(Teuchos::rcp(&Amat, false));
                auto info = Amat_solver.invert();
                //auto Amat_inv_ptr = Amat_solver.getFactoredMatrix(); //not needed
                //Amat is now its inverse
                
                //MA 200413 fix: basis_t6_.. should be squared according to definition of tau in VMSDG 
                const double tempfactor = std::pow(basis_T6_edge_values_pow_integrated[i](k, edge_ordinal), 2.0) / element_edge_lengths[i](k);

                for (int j1 = 0; j1 < _ndim_requested; ++j1){
                    for (int j2 = 0; j2 < _ndim_requested; ++j2) {
                        _tau(k, i, j1, j2) = Amat(j1, j2) * tempfactor;
                    }
                }
            } //loop over num_cells_local
        } //loop over num_element_sides
    }


    
    // get all quadrature put together here as auxiliary evaluation sites
    // store calculated quadrature values over fields values
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_cells_local), KOKKOS_LAMBDA(const int i) {
        for (int j=0; j<_weights_ndim; ++j) {
            if (j<num_element_cub_points) {
                quadrature_type(i,j) = 1;
                quadrature_weights(i,j) = physical_element_cub_weights(i,j);
                for (int k=0; k<_ndim_requested; ++k) {
                    quadrature_points(i,_ndim_requested*j+k) = physical_element_cub_points(i,j,k);
                }
            } 
            else {
                int current_side_num = (j - num_element_cub_points)/num_side_cub_points;
                quadrature_type(i,j) = ((int)adjacent_elements(i,current_side_num)>-1) ? 0 : 2;
                int local_cub_num = (j-num_element_cub_points)%num_side_cub_points;
                quadrature_weights(i,j) = physical_side_cub_weights[current_side_num](i, local_cub_num);
                for (int k=0; k<element_dim; ++k) {
                    quadrature_points(i,_ndim_requested*j+k) = physical_side_cub_points[current_side_num](i, local_cub_num, k);
                    unit_normals(i,_ndim_requested*j+k) = side_unit_normals(i, current_side_num, k);
                }
            }
        }
    });
    Kokkos::fence();
    _cells->getFieldManager()->updateFieldsHaloData();

    Kokkos::View<double**> kokkos_quadrature_coordinates("all quadrature coordinates", _cells->getCoordsConst()->nLocal()*_weights_ndim, 3);
    _kokkos_quadrature_coordinates_host = Kokkos::create_mirror_view(kokkos_quadrature_coordinates);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_cells_local), KOKKOS_LAMBDA(const int i) {
        for (int j=0; j<_weights_ndim; ++j) {
            for (int k=0; k<_ndim_requested; ++k) {
                _kokkos_quadrature_coordinates_host(i*_weights_ndim + j, k) = quadrature_points(i,_ndim_requested*j+k);
            }
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
    if (_velocity_basis_type_divfree) {
        _vel_gmls = Teuchos::rcp<GMLS>(new GMLS(DivergenceFreeVectorTaylorPolynomial,
                     VectorPointSample,
                     _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                     _ndim_requested,
                     _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"), 
                     "STANDARD", "NO_CONSTRAINT"));
    } else if (_velocity_basis_type_vector) {
        _vel_gmls = Teuchos::rcp<GMLS>(new GMLS(VectorTaylorPolynomial,
                     VectorPointSample,
                     _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                     _ndim_requested,
                     _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"), 
                     "STANDARD", "NO_CONSTRAINT"));
    } else {
        _vel_gmls = Teuchos::rcp<GMLS>(new GMLS(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                     _ndim_requested,
                     _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"), 
                     "STANDARD", "NO_CONSTRAINT"));
    }
    _vel_gmls->setProblemData(_kokkos_neighbor_lists_host,
                    _kokkos_augmented_source_coordinates_host,
                    _kokkos_target_coordinates_host,
                    _kokkos_epsilons_host);
    _vel_gmls->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
    _vel_gmls->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));

    if (_use_vector_gmls)
        _vel_gmls->addTargets(TargetOperation::VectorPointEvaluation);
    else
        _vel_gmls->addTargets(TargetOperation::ScalarPointEvaluation);
    if (_use_vector_grad_gmls)
        _vel_gmls->addTargets(TargetOperation::GradientOfVectorPointEvaluation);
    else
        _vel_gmls->addTargets(TargetOperation::GradientOfScalarPointEvaluation);

    _vel_gmls->setAdditionalEvaluationSitesData(_kokkos_quadrature_neighbor_lists_host, _kokkos_quadrature_coordinates_host);
    _vel_gmls->generateAlphas();

    if (_st_op || _mix_le_op) {
        _pressure_gmls = Teuchos::rcp<GMLS>(new GMLS(_parameters->get<Teuchos::ParameterList>("remap").get<int>("pressure porder"),
                     _ndim_requested,
                     _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"), 
                     "STANDARD", "NO_CONSTRAINT"));
        _pressure_gmls->setProblemData(_kokkos_neighbor_lists_host,
                        _kokkos_augmented_source_coordinates_host,
                        _kokkos_target_coordinates_host,
                        _kokkos_epsilons_host);
        _pressure_gmls->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
        _pressure_gmls->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));

        _pressure_gmls->addTargets(TargetOperation::ScalarPointEvaluation);
        _pressure_gmls->setAdditionalEvaluationSitesData(_kokkos_quadrature_neighbor_lists_host, _kokkos_quadrature_coordinates_host);
        _pressure_gmls->generateAlphas();
    }


    // halo
    auto nhalo = _cells->getCoordsConst()->nLocal(true)-num_cells_local;

    if (nhalo > 0) {

        // halo version of _cell_particles_neighborhood
        {
            // quantities contained on cells (mesh)
            auto halo_quadrature_points = _cells->getFieldManager()->getFieldByName("quadrature_points")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
            auto halo_quadrature_weights = _cells->getFieldManager()->getFieldByName("quadrature_weights")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
            auto halo_quadrature_type = _cells->getFieldManager()->getFieldByName("interior")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
            auto halo_unit_normals = _cells->getFieldManager()->getFieldByName("unit_normal")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
            
            // this quantity refers to global elements, so it's local number can be found by getting neighbor with same global number
            auto halo_adjacent_elements = _cells->getFieldManager()->getFieldByName("adjacent_elements")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();

            auto halo_needed = host_view_local_index_type("halo index needed", nhalo, 1);
            Kokkos::deep_copy(halo_needed, 0);

            // marks all neighbors reachable from locally owned cells
            for(local_index_type i = 0; i < num_cells_local; i++) {
                local_index_type num_neighbors = _cell_particles_neighborhood->getNumNeighbors(i);
                for (local_index_type l = 0; l < num_neighbors; l++) {
                    if (_cell_particles_neighborhood->getNeighbor(i,l) >= num_cells_local) {
                       auto index = _cell_particles_neighborhood->getNeighbor(i,l)-num_cells_local;
                       halo_needed(index,0) = 1;
                    }
                }
            }
            Kokkos::fence();

            _halo_big_to_small = host_view_local_index_type("halo big to small map", nhalo, 1);
            Kokkos::deep_copy(_halo_big_to_small, halo_needed);
            Kokkos::fence();

            //// DIAGNOSTIC:: check to see which DOFs are needed
            //if (_cell_particles_neighborhood->getSourceCoordinates()->getComm()->getRank()==3) {
            //    for (int i=0; i<nhalo; ++i) {
            //        printf("h:%d, %d\n", i, _halo_big_to_small(i,0));
            //    }
            //}

            Kokkos::parallel_scan(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nhalo), KOKKOS_LAMBDA(const int i, local_index_type& upd, const bool& final) {
                const local_index_type this_i = _halo_big_to_small(i,0);
                upd += this_i;
                if (final) {
                    _halo_big_to_small(i,0) = upd;
                }
            });
            Kokkos::fence();

            auto max_needed_entries = _halo_big_to_small(nhalo-1,0);
            _halo_small_to_big = host_view_local_index_type("halo small to big map", max_needed_entries, 1);

            //// DIAGNOSTIC:: check mappings to see results of scan
            //if (_cell_particles_neighborhood->getSourceCoordinates()->getComm()->getRank()==3) {
            //    for (int i=0; i<nhalo; ++i) {
            //        printf("a:%d, %d\n", i, _halo_big_to_small(i,0));
            //    }
            //}

            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nhalo), KOKKOS_LAMBDA(const int i) {
                _halo_big_to_small(i,0) *= halo_needed(i,0);
                _halo_big_to_small(i,0)--;
                if (_halo_big_to_small(i,0) >= 0) _halo_small_to_big(_halo_big_to_small(i,0),0) = i;
            });
            Kokkos::fence();
            //// DIAGNOSTIC:: check mappings from big to small and small to big to ensure they are consistent
            //if (_cell_particles_neighborhood->getSourceCoordinates()->getComm()->getRank()==3) {
            //    for (int i=0; i<nhalo; ++i) {
            //        printf("b:%d, %d\n", i, _halo_big_to_small(i,0));
            //    }
            //    for (int i=0; i<max_needed_entries; ++i) {
            //        printf("s:%d, %d\n", i, _halo_small_to_big(i,0));
            //    }
            //}

            Kokkos::View<double**> kokkos_halo_target_coordinates("halo target_coordinates", max_needed_entries, target_coords->nDim());
            auto kokkos_halo_target_coordinates_host = Kokkos::create_mirror_view(kokkos_halo_target_coordinates);
            // fill in the target, adding regular coordiantes only into a kokkos view
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,max_needed_entries), KOKKOS_LAMBDA(const int i) {
                xyz_type coordinate = target_coords->getLocalCoords(num_cells_local+_halo_small_to_big(i,0), true /*include halo*/, use_physical_coords);
                kokkos_halo_target_coordinates_host(i,0) = coordinate.x;
                kokkos_halo_target_coordinates_host(i,1) = coordinate.y;
                if (_ndim_requested>2) kokkos_halo_target_coordinates_host(i,2) = coordinate.z;
            });

            // make temporary particle set of halo target coordinates needed
            Teuchos::RCP<Compadre::ParticlesT> halo_particles =
                Teuchos::rcp( new Compadre::ParticlesT(_parameters, _cells->getCoordsConst()->getComm(), _parameters->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions")));
            halo_particles->resize(max_needed_entries, true);
            auto halo_coords = halo_particles->getCoords();
            //std::vector<Compadre::XyzVector> verts_to_insert(max_needed_entries);
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,max_needed_entries), KOKKOS_LAMBDA(const int i) {
                XyzVector coord_i(0,0,0);
                for (local_index_type j=0; j<_ndim_requested; ++j) {
                    coord_i[j] = kokkos_halo_target_coordinates_host(i,j);
                }
                halo_coords->replaceLocalCoords(i, coord_i, true);
                halo_coords->replaceLocalCoords(i, coord_i, false);
                //for (local_index_type j=0; j<_ndim_requested; ++j) {
                //    (*const_cast<Compadre::XyzVector*>(&verts_to_insert[i]))[j] = kokkos_halo_target_coordinates_host(i,j);
                //}
            });
            //halo_coords->insertCoords(verts_to_insert);
            //halo_particles->resetWithSameCoords();

            // DIAGNOSTIC:: pull coords from halo_particles and compare to what was inserted
            // auto retrieved_halo_coords = halo_particles->getCoords()->getPts()->getLocalView<const host_view_type>();
            // for (int i=0;i<max_needed_entries;++i) {
            //     printf("%.16f %.16f %.16f\n", retrieved_halo_coords(i,0)-kokkos_halo_target_coordinates_host(i,0), retrieved_halo_coords(i,0),kokkos_halo_target_coordinates_host(i,0));
            // }
            _cell_particles_neighborhood->getSourceCoordinates()->getComm()->barrier();
 
            // perform neighbor search to halo particles using same h as previous radii search
            _halo_cell_particles_neighborhood = Teuchos::rcp_static_cast<neighborhood_type>(Teuchos::rcp(
                    new neighborhood_type(_cells, halo_particles.getRawPtr(), false /*material coords*/, maxLeaf)));
            _halo_cell_particles_neighborhood->constructAllNeighborLists(max_search_size /* we know we have enough halo info */,
                "radius",
                true /*dry run for sizes*/,
                neighbors_needed+1,
                0.0, /* cutoff multiplier */
                _cell_particles_max_h, /* search size */
                false, /* uniform radii is true, but will be enforce automatically because all search sizes are the same globally */
                _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));
            Kokkos::fence();

            //// DIAGNOSTIC:: check that neighbors are actually close to halo target sites after search
            //for (int i=0;i<max_needed_entries;++i) {
            //    printf("i:%d, max: %.16f\n", i, max_h);
            //    for (int j=0; j<_halo_cell_particles_neighborhood->getNumNeighbors(i); ++j) {
            //        auto this_j = _cells->getCoordsConst()->getLocalCoords(_halo_cell_particles_neighborhood->getNeighbor(i,j), true, use_physical_coords);//_halo_small_to_big(j,0),i,j));
            //        auto diff = std::sqrt(pow(kokkos_halo_target_coordinates_host(i,0)-this_j[0],2) + pow(kokkos_halo_target_coordinates_host(i,1)-this_j[1],2));
            //        if (diff > max_h) {
            //            printf("(%d,%d,%d):: diff %.16f\n", i, j, _cell_particles_neighborhood->getSourceCoordinates()->getComm()->getRank(), diff);
            //            printf("s:%f %f %f\n", kokkos_halo_target_coordinates_host(i,0),kokkos_halo_target_coordinates_host(i,1),kokkos_halo_target_coordinates_host(i,2));
            //            printf("n:%f %f %f\n", this_j[0],this_j[1],this_j[2]);//alo_target_coordinates_host(i,0),kokkos_halo_target_coordinates_host(i,1),kokkos_halo_target_coordinates_host(i,2));
            //        }
            //    }
            //}

            // extract neighbor search data into a view
            auto halo_cell_particles_max_num_neighbors = _halo_cell_particles_neighborhood->computeMaxNumNeighbors(false /* local processor max*/);
            Kokkos::fence();
            Kokkos::View<int**> kokkos_halo_neighbor_lists("halo neighbor lists", max_needed_entries, 
                    halo_cell_particles_max_num_neighbors+1);
            auto kokkos_halo_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_halo_neighbor_lists);
            // fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,max_needed_entries), KOKKOS_LAMBDA(const int i) {
                const int num_i_neighbors = _halo_cell_particles_neighborhood->getNumNeighbors(i);
                for (int j=1; j<num_i_neighbors+1; ++j) {
                    kokkos_halo_neighbor_lists_host(i,j) = _halo_cell_particles_neighborhood->getNeighbor(i,j-1);
                }
                kokkos_halo_neighbor_lists_host(i,0) = num_i_neighbors;
            });

            auto halo_epsilons = _halo_cell_particles_neighborhood->getHSupportSizes()->getLocalView<const host_view_type>();
            Kokkos::View<double*> kokkos_halo_epsilons("halo epsilons", halo_epsilons.extent(0), 1);
            auto kokkos_halo_epsilons_host = Kokkos::create_mirror_view(kokkos_halo_epsilons);
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,halo_epsilons.extent(0)), KOKKOS_LAMBDA(const int i) {
                kokkos_halo_epsilons_host(i) = halo_epsilons(i,0);
            });

            // get all quadrature put together here as auxiliary evaluation site
            Kokkos::View<double**> kokkos_halo_quadrature_coordinates("halo all quadrature coordinates", 
                    max_needed_entries*_weights_ndim, 3);
            auto kokkos_halo_quadrature_coordinates_host = Kokkos::create_mirror_view(kokkos_halo_quadrature_coordinates);
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,max_needed_entries), 
                    KOKKOS_LAMBDA(const int i) {
                for (int j=0; j<_weights_ndim; ++j) {
                    for (int k=0; k<_ndim_requested; ++k) {
                        kokkos_halo_quadrature_coordinates_host(i*_weights_ndim + j, k) = 
                            halo_quadrature_points(_halo_small_to_big(i,0),_ndim_requested*j+k);
                    }
                }
            });

            // get cell neighbors of particles, and add indices to these for quadrature
            // loop over particles, get cells in neighborhood and get their quadrature
            Kokkos::View<int**> kokkos_halo_quadrature_neighbor_lists("halo quadrature neighbor lists", max_needed_entries, _weights_ndim+1);
            auto kokkos_halo_quadrature_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_halo_quadrature_neighbor_lists);
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,max_needed_entries), 
                    KOKKOS_LAMBDA(const int i) {
                kokkos_halo_quadrature_neighbor_lists_host(i,0) = static_cast<int>(_weights_ndim);
                for (int j=0; j<kokkos_halo_quadrature_neighbor_lists(i,0); ++j) {
                    kokkos_halo_quadrature_neighbor_lists_host(i, 1+j) = i*_weights_ndim + j;
                }
            });

            // need _kokkos_halo_quadrature_neighbor lists and quadrature coordinates
            // GMLS operator
            _halo_vel_gmls = Teuchos::rcp<GMLS>(new GMLS(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                         _ndim_requested /* dimension */,
                         _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"), 
                         "STANDARD", "NO_CONSTRAINT"));
            _halo_vel_gmls->setProblemData(kokkos_halo_neighbor_lists_host,
                            _kokkos_augmented_source_coordinates_host,
                            kokkos_halo_target_coordinates_host,
                            kokkos_halo_epsilons_host);
            _halo_vel_gmls->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
            _halo_vel_gmls->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));

            _halo_vel_gmls->addTargets(TargetOperation::ScalarPointEvaluation);
            _halo_vel_gmls->addTargets(TargetOperation::GradientOfScalarPointEvaluation);
            _halo_vel_gmls->setAdditionalEvaluationSitesData(kokkos_halo_quadrature_neighbor_lists_host, kokkos_halo_quadrature_coordinates_host);
            _halo_vel_gmls->generateAlphas();
            Kokkos::fence();
        }
    }


    GenerateData->stop();

#endif // TRILINOS_DISCRETIZATION
}

local_index_type ReactionDiffusionPhysics::getMaxNumNeighbors() {
    // _particles_triple_hop_neighborhood set to null after computeGraph is called
    TEUCHOS_ASSERT(!_particles_triple_hop_neighborhood.is_null());
    return _particles_triple_hop_neighborhood->computeMaxNumNeighbors(false /*local processor maximum*/);
}

Kokkos::View<size_t*, Kokkos::HostSpace> ReactionDiffusionPhysics::getMaxEntriesPerRow(local_index_type field_one, local_index_type field_two) {

    auto comm = this->_particles->getCoordsConst()->getComm();

    const local_index_type num_particles_local = static_cast<local_index_type>(this->_coords->nLocal());
    Kokkos::View<size_t*, Kokkos::HostSpace> maxEntriesPerRow("max entries per row", num_particles_local);

    const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
    const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();

    if (field_one == _lagrange_field_id && field_two == _pressure_field_id) {
        auto row_map_entries = _row_map->getMyGlobalIndices();
        Kokkos::resize(maxEntriesPerRow, row_map_entries.extent(0));
        Kokkos::deep_copy(maxEntriesPerRow, 0);
        if (comm->getRank()==0) {
            maxEntriesPerRow(0) = num_particles_local*fields[field_two]->nDim();
        } else {
            maxEntriesPerRow(row_map_entries.extent(0)-1) = num_particles_local*fields[field_two]->nDim();
        }
        // write to last entry
    } else if (field_one == _pressure_field_id && field_two == _lagrange_field_id) {
        for(local_index_type i = 0; i < num_particles_local; i++) {
            maxEntriesPerRow(i) = fields[field_two]->nDim();
        }
    }

    return maxEntriesPerRow;
}

Teuchos::RCP<crs_graph_type> ReactionDiffusionPhysics::computeGraph(local_index_type field_one, local_index_type field_two) {
    if (field_two == -1) {
        field_two = field_one;
    }

    const local_index_type num_particles_local = static_cast<local_index_type>(this->_coords->nLocal());

    Teuchos::RCP<Teuchos::Time> ComputeGraphTime = Teuchos::TimeMonitor::getNewCounter ("Compute Graph Time");
    ComputeGraphTime->start();
    TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A_graph.is_null(), "Tpetra CrsGraph for Physics not yet specified.");

    if (field_one == _lagrange_field_id && field_two == _pressure_field_id) {
        // row all DOFs for solution against Lagrange Multiplier

        // create a new graph from the existing one, but with an updated row map augmented by a single global dof
        // for the lagrange multiplier
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

        for (size_t i=0; i<row_map_entries.extent(0); ++i) {
            //new_row_map_entries(i+offset_size) = row_map_entries(i);
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
                // now the first entry in the map is to the shared gid for the LM
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

    } else if (field_one == _pressure_field_id && field_two == _lagrange_field_id) {

        // create a new graph from the existing one, but with an updated col map augmented by a single global dof
        // for the lagrange multiplier
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

        for (size_t i=0; i<col_map_entries.extent(0); ++i) {
            //new_col_map_entries(i+offset_size) = col_map_entries(i);
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
                // now the first entry in the map is to the shared gid for the LM
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

    const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    if (((field_one == _velocity_field_id) || (field_one == _pressure_field_id)) && ((field_two == _velocity_field_id) || (field_two == _pressure_field_id))) {
        for(local_index_type i = 0; i < num_particles_local; i++) {
            local_index_type num_neighbors = _particles_triple_hop_neighborhood->getNumNeighbors(i);
            for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
                local_index_type row = local_to_dof_map(i, field_one, k);

                Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
                Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

                for (local_index_type l = 0; l < num_neighbors; l++) {
                    for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                        col_data[l*fields[field_two]->nDim() + n] = local_to_dof_map(_particles_triple_hop_neighborhood->getNeighbor(i,l), field_two, n);
                    }
                }
                {
                    this->_A_graph->insertLocalIndices(row, cols);
                }
            }
        }
    } else if (field_one == _lagrange_field_id && field_two == _pressure_field_id) {
        // row all DOFs for solution against Lagrange Multiplier
        Teuchos::Array<local_index_type> col_data(num_particles_local * fields[field_two]->nDim());
        Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

        for (local_index_type l = 0; l < num_particles_local; l++) {
            for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                cols[l*fields[field_two]->nDim() + n] = local_to_dof_map(l, field_two, n);
            }
        }
        // local index 0 is global index shared by all processors
        //local_index_type row = local_to_dof_map[0][field_one][0];
        //this->_A_graph->insertLocalIndices(0, cols);

        auto comm = this->_particles->getCoordsConst()->getComm();
        if (comm->getRank() == 0) {
            // local index 0 is the shared global id for the lagrange multiplier
            this->_A_graph->insertLocalIndices(0, cols);
        } else {
            // local index 0 is the shared global id for the lagrange multiplier
            this->_A_graph->insertLocalIndices(num_particles_local, cols);
        }

    } else if (field_one == _pressure_field_id && field_two == _lagrange_field_id) {
        // col all DOFs for solution against Lagrange Multiplier
        auto comm = this->_particles->getCoordsConst()->getComm();
        //if (comm->getRank() == 0) {
            for(local_index_type i = 0; i < num_particles_local; i++) {
                for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
                    local_index_type row = local_to_dof_map(i, field_one, k);

                    Teuchos::Array<local_index_type> col_data(1);
                    Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
                    if (comm->getRank() == 0) {
                        cols[0] = 0; // local index 0 is global shared index
                    } else {
                        cols[0] = num_particles_local; // local index 0 is global shared index
                    }
                    this->_A_graph->insertLocalIndices(row, cols);
                }
            }
        //}
    } else if (field_one == _lagrange_field_id && field_two == _lagrange_field_id) {
        // identity on all DOFs for Lagrange Multiplier (even DOFs not really used, since we only use first LM dof for now)
        for(local_index_type i = 0; i < num_particles_local; i++) {
            for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
                local_index_type row = local_to_dof_map(i, field_one, k);

                Teuchos::Array<local_index_type> col_data(fields[field_two]->nDim());
                Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
                for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                    cols[n] = local_to_dof_map(i, field_two, n);
                }
                //#pragma omp critical
                {
                    this->_A_graph->insertLocalIndices(row, cols);
                }
            }
        }
    }
    // set neighborhood to null because it is large (storage) and not used again
    //_particles_triple_hop_neighborhood = Teuchos::null;
    ComputeGraphTime->stop();
    return this->_A_graph;
}


void ReactionDiffusionPhysics::computeMatrix(local_index_type field_one, local_index_type field_two, scalar_type time) {

    bool use_physical_coords = true; // can be set on the operator in the future

    if (!_l2_op) {
        TEUCHOS_ASSERT(_cells->getCoordsConst()->getComm()->getSize()==1);
    }

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

    auto halo_quadrature_points = _cells->getFieldManager()->getFieldByName("quadrature_points")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_quadrature_weights = _cells->getFieldManager()->getFieldByName("quadrature_weights")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_quadrature_type = _cells->getFieldManager()->getFieldByName("interior")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_unit_normals = _cells->getFieldManager()->getFieldByName("unit_normal")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_adjacent_elements = _cells->getFieldManager()->getFieldByName("adjacent_elements")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();


    auto penalty = (_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")+1)*_parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_cell_particles_max_h;
    printf("Computed penalty parameter: %.16f\n", penalty); 
    //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")*_particles_particles_neighborhood->getMinimumHSupportSize();

    // each element must have the same number of sides
    auto num_sides = adjacent_elements.extent(1);
    auto num_interior_quadrature = 0;
    for (int q=0; q<_weights_ndim; ++q) {
        if (quadrature_type(0,q)!=1) { // some type of side quadrature
            num_interior_quadrature = q;
            break;
        }
    }
    auto num_exterior_quadrature_per_side = (_weights_ndim - num_interior_quadrature)/num_sides;

    double area = 0; // (good, no rewriting)
    double perimeter = 0;

    //double t_area = 0; 
    //double t_perimeter = 0;

    // this maps should be for nLocal(true) so that it contains local neighbor lookups for owned and halo particles
    std::vector<std::unordered_map<local_index_type, local_index_type> > particle_to_local_neighbor_lookup(_cells->getCoordsConst()->nLocal(true), std::unordered_map<local_index_type, local_index_type>());
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_cells->getCoordsConst()->nLocal(true)), [&](const int i) {
    //for (int i=0; i<_cells->getCoordsConst()->nLocal(true); ++i) {
        std::unordered_map<local_index_type, local_index_type>* i_map = const_cast<std::unordered_map<local_index_type, local_index_type>* >(&particle_to_local_neighbor_lookup[i]);
        auto halo_i = (i<nlocal) ? -1 : _halo_big_to_small(i-nlocal,0);
        if (i>=nlocal /* halo */  && halo_i<0 /* not one found within max_h of locally owned particles */) return;
        local_index_type num_neighbors = (i<nlocal) ? _cell_particles_neighborhood->getNumNeighbors(i)
            : _halo_cell_particles_neighborhood->getNumNeighbors(halo_i);
        for (local_index_type j=0; j<num_neighbors; ++j) {
            auto particle_j = (i<nlocal) ? _cell_particles_neighborhood->getNeighbor(i,j)
                : _halo_cell_particles_neighborhood->getNeighbor(halo_i,j);
            (*i_map)[particle_j]=j;
        }
    });
    Kokkos::fence();

    

    scalar_type pressure_coeff = 1.0;
    if (_mix_le_op) {
        pressure_coeff = _lambda + 2./(scalar_type)(_ndim_requested)*_shear;
    }

    if (((field_one == _velocity_field_id) || (field_one == _pressure_field_id)) && ((field_two == _velocity_field_id) || (field_two == _pressure_field_id))) {
    // loop over cells including halo cells 
    //Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_cells->getCoordsConst()->nLocal(true)), [&](const int i, scalar_type& t_area) {
    //for (int i=0; i<_cells->getCoordsConst()->nLocal(true); ++i) {
    int team_scratch_size = 0;
    team_scratch_size += host_scratch_vector_scalar_type::shmem_size(num_sides); // side lengths
    team_scratch_size += host_scratch_vector_scalar_type::shmem_size(5*_weights_ndim); // indices
    Kokkos::parallel_for("Assembly Routine", host_team_policy(_cells->getCoordsConst()->nLocal(true), Kokkos::AUTO)
            .set_scratch_size(1 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
            KOKKOS_LAMBDA(const host_member_type& teamMember) {
 
        const int i = teamMember.league_rank();

        host_scratch_vector_scalar_type side_lengths(teamMember.team_scratch(0 /*shared memory*/), num_sides);
        host_scratch_vector_local_index_type current_side_num(teamMember.team_scratch(0 /*shared memory*/), _weights_ndim);
        host_scratch_vector_local_index_type side_of_cell_i_to_adjacent_cell(teamMember.team_scratch(0 /*shared memory*/), _weights_ndim);
        host_scratch_vector_local_index_type adjacent_cell_local_index(teamMember.team_scratch(0 /*shared memory*/), _weights_ndim);
        host_scratch_vector_local_index_type j_to_adjacent_cell(teamMember.team_scratch(0 /*shared memory*/), _weights_ndim);
        host_scratch_vector_local_index_type k_to_adjacent_cell(teamMember.team_scratch(0 /*shared memory*/), _weights_ndim);

        //// TODO: REMOVE LATER!
        //if (_pressure_field_id==field_one || _pressure_field_id==field_two) return;

        std::unordered_set<local_index_type> cell_neighbors;

        double val_data[1] = {0};
        int    col_data[1] = {-1};

        double t_perimeter = 0; 

        // filter out (skip) all halo cells that are not able to be seen from a locally owned particle
        // dictionary needs created of halo_particle
        // halo_big_to_small 
        auto halo_i = (i<nlocal) ? -1 : _halo_big_to_small(i-nlocal,0);
        if (i>=nlocal /* halo */  && halo_i<0 /* not one found within max_h of locally owned particles */) return;//continue;

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

        // DIAGNOSTIC:: sum area over locally owned cells

        //std::vector<double> side_lengths(num_sides);
        {
            int current_side_num = 0;
            for (int q=0; q<_weights_ndim; ++q) {
                auto q_type = (i<nlocal) ? quadrature_type(i,q) 
                    : halo_quadrature_type(_halo_small_to_big(halo_i,0),q);
                if (q_type!=1) { // side
                    auto q_wt = (i<nlocal) ? quadrature_weights(i,q) 
                        : halo_quadrature_weights(_halo_small_to_big(halo_i,0),q);
                    int new_current_side_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_side;
                    if (new_current_side_num!=current_side_num) {
                        side_lengths[new_current_side_num] = q_wt;
                        current_side_num = new_current_side_num;
                    } else {
                        side_lengths[current_side_num] += q_wt;
                    }
                }
            }
        }

        // information needed for calculating jump terms
        //std::vector<int> current_side_num(_weights_ndim);
        //std::vector<int> side_of_cell_i_to_adjacent_cell(_weights_ndim);
        //std::vector<int> adjacent_cell_local_index(_weights_ndim);
        if (!_l2_op) {
            TEUCHOS_ASSERT(_cells->getCoordsConst()->getComm()->getSize()==1);
            for (int q=num_interior_quadrature; q<_weights_ndim; ++q) {
                current_side_num[q] = (q - num_interior_quadrature)/num_exterior_quadrature_per_side;
                adjacent_cell_local_index[q] = (int)(adjacent_elements(i, current_side_num[q]));
                if (adjacent_cell_local_index[q]>-1) {
                    side_of_cell_i_to_adjacent_cell[q] = -1;
                    for (int z=0; z<num_sides; ++z) {
                        auto adjacent_cell_to_adjacent_cell = (int)(adjacent_elements(adjacent_cell_local_index[q],z));
                        if (adjacent_cell_to_adjacent_cell==i) {
                            side_of_cell_i_to_adjacent_cell[q] = z;
                            break;
                        }
                    }
                }
            }
            // not yet set up for MPI
            //for (int q=0; q<_weights_ndim; ++q) {
            //    if (q>=num_interior_quadrature) {
            //        current_side_num[q] = (q - num_interior_quadrature)/num_exterior_quadrature_per_side;
            //        adjacent_cell_local_index[q] = (i<nlocal) ? (int)(adjacent_elements(i, current_side_num[q]))
            //            : (int)(halo_adjacent_elements(halo_i, current_side_num[q]));
            //        side_of_cell_i_to_adjacent_cell[q] = -1;
            //        for (int z=0; z<num_exterior_quadrature_per_side; ++z) {
            //            auto adjacent_cell_to_adjacent_cell = (adjacent_cell_local_index[q]<nlocal) ?
            //                (int)(adjacent_elements(adjacent_cell_local_index[q],z))
            //                : (int)(halo_adjacent_elements(adjacent_cell_local_index[q]-nlocal,z));
            //            if (adjacent_cell_to_adjacent_cell==i) {
            //                side_of_cell_i_to_adjacent_cell[q] = z;
            //                break;
            //            }
            //        }
            //    }
            //}
        }

//        //std::vector<int> adjacent_cells(3,-1); // initialize all entries to -1
//        //int this_side_num = -1;
//        //for (int q=0; q<_weights_ndim; ++q) {
//        //    int new_side_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_side;
//        //    if (new_side_num != this_side_num) {
//        //        this_side_num = new_side_num;
//        //        adjacent_cells[this_side_num] = (int)(adjacent_elements(i, this_side_num));
//        //    }
//        //}
//        //TEUCHOS_ASSERT(this_side_num==2);
//
//        int this_side_num = -1;
//        for (int q=0; q<_weights_ndim; ++q) {
//            this_side_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_side;
//            //printf("q: %d, e: %d\n", q, this_side_num);
//            if (i<nlocal) { // locally owned
//                if (quadrature_type(i,q)==1) { // interior
//                    //t_area += quadrature_weights(i,q);
//                    auto x=quadrature_points(i,2*q+0);
//                    auto y=quadrature_points(i,2*q+1);
//                    t_area += quadrature_weights(i,q) * (2 + 3*x + 3*y);
//                } else if (quadrature_type(i,q)==2) { // exterior quad point on exterior side
//                    //t_perimeter += quadrature_weights(i,q);
//                    //t_perimeter += quadrature_weights(i,q ) * (quadrature_points(i,2*q+0)*quadrature_points(i,2*q+0) + quadrature_points(i,2*q+1)*quadrature_points(i,2*q+1));
//                    // integral of x^2 + y^2 over (0,0),(2,0),(2,2),(0,2) is
//                    // 4*8/3+2*8 = 26.666666
//                    auto x=quadrature_points(i,2*q+0);
//                    auto y=quadrature_points(i,2*q+1);
//                    auto nx=unit_normals(i,2*q+0);
//                    auto ny=unit_normals(i,2*q+1);
//                    t_perimeter += quadrature_weights(i,q) * (x+x*x+x*y+y+y*y) * (nx + ny);
//                } else if (quadrature_type(i,q)==0) { // interior quad point on exterior side
//                    //if (i>adjacent_elements(i,this_side_num)) {
//                    //    t_perimeter += quadrature_weights(i,q ) * 1 / side_lengths[this_side_num];
//                    //    if (q%3==0) num_interior_sides++;
//                    //}
//                    //auto x=quadrature_points(i,2*q+0);
//                    //auto y=quadrature_points(i,2*q+1);
//                    //auto nx=unit_normals(i,2*q+0);
//                    //auto ny=unit_normals(i,2*q+1);
//                    int adjacent_cell = adjacent_elements(i,this_side_num);
//                    int side_i_to_adjacent_cell = -1;
//                    for (int z=0; z<3; ++z) {
//                        if ((int)adjacent_elements(adjacent_cell,z)==i) {
//                            side_i_to_adjacent_cell = z;
//                        }
//                    }
//                    
//                    int adjacent_q = num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_side + (num_exterior_quadrature_per_side - ((q-num_interior_quadrature)%num_exterior_quadrature_per_side) - 1);
//
//                    //auto x=quadrature_points(adjacent_cell, num_interior_quadrature + 2*(side_i_to_adjacent_cell+(q%3))+0);
//                    //auto y=quadrature_points(adjacent_cell, num_interior_quadrature + 2*(side_i_to_adjacent_cell+(q%3))+1);
//                    //auto x=quadrature_points(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_side+((q-num_interior_quadrature)%num_exterior_quadrature_per_side))+0);
//                    //auto y=quadrature_points(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_side+((q-num_interior_quadrature)%num_exterior_quadrature_per_side))+1);
//                    auto x=quadrature_points(adjacent_cell, 2*adjacent_q+0);
//                    auto y=quadrature_points(adjacent_cell, 2*adjacent_q+1);
//
//                    //auto y=quadrature_points(adjacent_cell,2*q+1);
//                    //
//                    auto nx=-unit_normals(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_side+((q-num_interior_quadrature)%num_exterior_quadrature_per_side))+0);
//                    auto ny=-unit_normals(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_side+((q-num_interior_quadrature)%num_exterior_quadrature_per_side))+1);
//                    //auto nx=unit_normals(i,2*q+0);
//                    //auto ny=unit_normals(i,2*q+1);
//                    t_perimeter += quadrature_weights(i,q) * (x+x*x+x*y+y+y*y) * (nx + ny);
//                }
//            }
//        }

        // creates a map that contains all neighbors from cell i plus any neighbors of cells
        // adjacent to cell i not contained in cell i's neighbor list
        {
            //local_index_type num_neighbors = _cell_particles_neighborhood->getNumNeighbors(i);
            local_index_type num_neighbors = (i<nlocal) ? _cell_particles_neighborhood->getNumNeighbors(i)
                : _halo_cell_particles_neighborhood->getNumNeighbors(halo_i);
            for (local_index_type j=0; j<num_neighbors; ++j) {
                //auto particle_j = _cell_particles_neighborhood->getNeighbor(i,j);
                auto particle_j = (i<nlocal) ? _cell_particles_neighborhood->getNeighbor(i,j)
                    : _halo_cell_particles_neighborhood->getNeighbor(halo_i,j);
                cell_neighbors.insert(particle_j);
            }
            if (!_l2_op) {
                TEUCHOS_ASSERT(_cells->getCoordsConst()->getComm()->getSize()==1);
                for (local_index_type j=0; j<num_sides; ++j) {
                    int adj_el = (int)(adjacent_elements(i,j));
                    if (adj_el>=0) {
                        num_neighbors = _cell_particles_neighborhood->getNumNeighbors(adj_el);
                        for (local_index_type k=0; k<num_neighbors; ++k) {
                            auto particle_k = _cell_particles_neighborhood->getNeighbor(adj_el,k);
                            cell_neighbors.insert(particle_k);
                        }
                    }
                }
                // not yet set up for MPI
                //for (local_index_type j=0; j<3; ++j) {
                //    int adj_el = (i<nlocal) ? (int)(adjacent_elements(i,j)) : (int)(halo_adjacent_elements(i-nlocal,j));
                //    if (adj_el>=0) {
                //        num_neighbors = (adj_el<nlocal) ? _cell_particles_neighborhood->getNumNeighbors(adj_el)
                //            : _halo_cell_particles_neighborhood->getNumNeighbors(adj_el);
                //        for (local_index_type k=0; k<num_neighbors; ++k) {
                //            auto particle_k = (adj_el<nlocal) ? _cell_particles_neighborhood->getNeighbor(adj_el,k)
                //                : _halo_cell_particles_neighborhood->getNeighbor(adj_el-nlocal,k);
                //            cell_neighbors[particle_k] = 1;
                //        }
                //    }
                //}
            }
        }

        local_index_type num_neighbors = cell_neighbors.size();
        //local_index_type num_neighbors = (i<nlocal) ? _particles_double_hop_neighborhood->getNumNeighbors(i)
        //    : _halo_particles_double_hop_neighborhood->getNumNeighbors(halo_i);

        int j=-1;
        //for (local_index_type j=0; j<num_neighbors; ++j) {
        for (auto j_it=cell_neighbors.begin(); j_it!=cell_neighbors.end(); ++j_it) {
            j++;
            auto particle_j = *j_it;
            //auto particle_j = (i<nlocal) ? _particles_double_hop_neighborhood->getNeighbor(i,j)
            //    : _halo_particles_double_hop_neighborhood->getNeighbor(halo_i,j);
            // particle_j is an index from {0..nlocal+nhalo}
            if (particle_j>=nlocal /* particle is halo particle */) continue; // row should be locally owned

            int j_to_cell_i = -1;
            if (particle_to_local_neighbor_lookup.at(i).count(particle_j)==1){
                j_to_cell_i = particle_to_local_neighbor_lookup.at(i).at(particle_j);
            }

            int j_has_value_from_adjacent=0;
            if (!_l2_op) {
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, _weights_ndim), [&] (const int qn, int& t_j_has_value_from_adjacent) {
                //for (int qn=0; qn<_weights_ndim; ++qn) {
                    const auto q_type = quadrature_type(i,qn);
                    if (q_type==0)  { // side on interior
                        j_to_adjacent_cell[qn] = -1;
                        const int adjacent_cell_local_index_q = adjacent_cell_local_index[qn];
                        //TEUCHOS_ASSERT(adjacent_cell_local_index_q >= 0);
                        if (i <= adjacent_cell_local_index_q) return;
                        if (particle_to_local_neighbor_lookup[adjacent_cell_local_index_q].count(particle_j)==1) {
                            j_to_adjacent_cell[qn] = particle_to_local_neighbor_lookup.at(adjacent_cell_local_index_q).at(particle_j);
                            t_j_has_value_from_adjacent++;
                        }
                    }
                }, j_has_value_from_adjacent);
            }

            int k=-1;
            //for (local_index_type k=0; k<num_neighbors; ++k) {
            for (auto k_it=cell_neighbors.begin(); k_it!=cell_neighbors.end(); ++k_it) {
                k++;
                auto particle_k = *k_it;
                //auto particle_k = (i<nlocal) ? _particles_double_hop_neighborhood->getNeighbor(i,k)
                //    : _halo_particles_double_hop_neighborhood->getNeighbor(halo_i,k);
                // particle_k is an index from {0..nlocal+nhalo}
                int k_to_cell_i = -1;
                if (particle_to_local_neighbor_lookup.at(i).count(particle_k)==1){
                    k_to_cell_i = particle_to_local_neighbor_lookup.at(i).at(particle_k);
                }


                // do filter out (skip) all halo particles that are not seen by local DOF
                // wrong thing to do, because k could be double hop through cell i
                int k_has_value_from_adjacent=0;
                if (!_l2_op) {
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, _weights_ndim), [&] (const int qn, int& t_k_has_value_from_adjacent) {
                    //for (int qn=0; qn<_weights_ndim; ++qn) {
                        const auto q_type = quadrature_type(i,qn);
                        if (q_type==0)  { // side on interior
                            k_to_adjacent_cell[qn] = -1;
                            const int adjacent_cell_local_index_q = adjacent_cell_local_index[qn];
                            //TEUCHOS_ASSERT(adjacent_cell_local_index_q >= 0);
                            if (i <= adjacent_cell_local_index_q) return;
                            if (particle_to_local_neighbor_lookup[adjacent_cell_local_index_q].count(particle_k)==1) {
                                k_to_adjacent_cell[qn] = particle_to_local_neighbor_lookup.at(adjacent_cell_local_index_q).at(particle_k);
                                t_k_has_value_from_adjacent++;
                            }
                        }
                    }, k_has_value_from_adjacent);
                    teamMember.team_barrier();
                }

                bool j_has_value = (j_to_cell_i>=0) || (j_has_value_from_adjacent>0);
                bool k_has_value = (k_to_cell_i>=0) || (k_has_value_from_adjacent>0);

                // locally owned DOF
                for (local_index_type j_comp_out = 0; j_comp_out < fields[field_one]->nDim(); ++j_comp_out) {
                for (local_index_type j_comp_in = 0; j_comp_in < fields[field_one]->nDim(); ++j_comp_in) {
                    if ((j_comp_out!=j_comp_in) && ((!_velocity_basis_type_vector && field_one == _velocity_field_id) || (field_one == _pressure_field_id))) continue;

                    local_index_type row = local_to_dof_map(particle_j, field_one, j_comp_in);

                    for (local_index_type k_comp_out = 0; k_comp_out < fields[field_two]->nDim(); ++k_comp_out) {
                    for (local_index_type k_comp_in = 0; k_comp_in < fields[field_two]->nDim(); ++k_comp_in) {
                        if ((k_comp_out!=k_comp_in) && ((!_velocity_basis_type_vector && field_two == _velocity_field_id) || (field_two == _pressure_field_id))) continue;

                        col_data[0] = local_to_dof_map(particle_k, field_two, k_comp_in);

                        double contribution = 0;
                        if (_l2_op) {
                            for (int qn=0; qn<_weights_ndim; ++qn) {
                                const auto q_type = (i<nlocal) ? quadrature_type(i,qn) : halo_quadrature_type(_halo_small_to_big(halo_i,0),qn);
                                if (q_type==1) { // interior
                                    double u, v;
                                    const auto q_wt = (i<nlocal) ? quadrature_weights(i,qn) : halo_quadrature_weights(_halo_small_to_big(halo_i,0),qn);
                                    if (j_to_cell_i>=0) {
                                        if (_use_vector_gmls) {
                                            v = (i<nlocal) ? _vel_gmls->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, j_comp_out, j_to_cell_i, j_comp_in, qn+1)
                                                 : _halo_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j_to_cell_i, qn+1);
                                        } else {
                                            v = (i<nlocal) ? _vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, qn+1)
                                                 : _halo_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j_to_cell_i, qn+1);
                                        }
                                    } else {
                                        v = 0;
                                    }
                                    if (k_to_cell_i>=0) {
                                        if (_use_vector_gmls) {
                                            u = (i<nlocal) ? _vel_gmls->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, k_comp_out, k_to_cell_i, k_comp_in, qn+1)
                                                 : _halo_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, k_to_cell_i, qn+1);
                                        } else {
                                            u = (i<nlocal) ? _vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, qn+1)
                                                 : _halo_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, k_to_cell_i, qn+1);
                                        }
                                    } else {
                                        u = 0;
                                    }

                                    contribution += q_wt * u * v;
                                } 
                                TEUCHOS_ASSERT(contribution==contribution);
                            }
                        } else {
                            // loop over quadrature
                            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, _weights_ndim), [&] (const int qn, double& t_contribution) {
                            //for (int qn=0; qn<_weights_ndim; ++qn) {
                                const auto q_type = quadrature_type(i,qn);
                                const auto q_wt = quadrature_weights(i,qn);
                                double u, v;
                                XYZ grad_u, grad_v;
                                
                                if (j_to_cell_i>=0) {

                                    if (_use_vector_gmls) v = _vel_gmls->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, j_comp_out, j_to_cell_i, j_comp_in, qn+1);
                                    else v = _vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, qn+1);

                                    if (_use_vector_grad_gmls) {
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            grad_v[d] = _vel_gmls->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, i, j_comp_out, d, j_to_cell_i, j_comp_in, qn+1);
                                        }
                                    } else {
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            grad_v[d] = _vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, d, j_to_cell_i, qn+1);
                                        }
                                    }
                                } else {
                                    v = 0.0;
                                    // grad_v set to 0's at instantiation
                                }
                                if (k_to_cell_i>=0) {

                                    if (_use_vector_gmls) u = _vel_gmls->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, k_comp_out, k_to_cell_i, k_comp_in, qn+1);
                                    else u = _vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, qn+1);

                                    if (_use_vector_grad_gmls) {
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            grad_u[d] = _vel_gmls->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, i, k_comp_out, d, k_to_cell_i, k_comp_in, qn+1);
                                        }
                                    } else {
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            grad_u[d] = _vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, d, k_to_cell_i, qn+1);
                                        }
                                    }
                                } else {
                                    u = 0.0;
                                    // grad_u set to 0's at instantiation
                                }

                                if (q_type==1) { // interior

                                    if (_rd_op) {
                                        t_contribution += _reaction * q_wt * u * v;
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            t_contribution += _diffusion * q_wt 
                                                * grad_v[d] * grad_u[d];
                                        }
                                    } else if (_le_op) {
                                        double div_v = 0, div_u = 0;
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            div_v += grad_v[d]*(j_comp_out==d);
                                            div_u += grad_u[d]*(k_comp_out==d);
                                        }
                                        t_contribution += q_wt * _lambda * div_v * div_u;
                                       
                                        double sigma_v_dot_sigma_u = 0.0;
                                        for (int d1=0; d1<_ndim_requested; ++d1) {
                                            for (int d2=0; d2<_ndim_requested; ++d2) {
                                                sigma_v_dot_sigma_u += (0.5*grad_v[d1]*(j_comp_out==d2)+0.5*grad_v[d2]*(j_comp_out==d1))*(0.5*grad_u[d1]*(k_comp_out==d2)+0.5*grad_u[d2]*(k_comp_out==d1));
                                            }
                                        }
                                        t_contribution += q_wt * 2 * _shear * sigma_v_dot_sigma_u;
                                        //(
                                        //      grad_v_x*grad_u_x*(j_comp_out==0)*(k_comp_out==0) 
                                        //    + grad_v_y*grad_u_y*(j_comp_out==1)*(k_comp_out==1) 
                                        //    + 0.5*(grad_v_y*(j_comp_out==0) + grad_v_x*(j_comp_out==1))*(grad_u_y*(k_comp_out==0) + grad_u_x*(k_comp_out==1)));
                                    } else if (_vl_op) {
                                        double grad_v_dot_grad_u = 0.0;
                                        for (int d1=0; d1<_ndim_requested; ++d1) {
                                            for (int d2=0; d2<_ndim_requested; ++d2) {
                                                grad_v_dot_grad_u += grad_v[d1]*(j_comp_out==d2)*grad_u[d1]*(k_comp_out==d2);
                                            }
                                        }
                                        t_contribution += q_wt * _shear* grad_v_dot_grad_u;
                                        //(
                                        //      grad_v_x*grad_u_x*(j_comp_out==0)*(k_comp_out==0) 
                                        //    + grad_v_y*grad_u_y*(j_comp_out==0)*(k_comp_out==0) 
                                        //    + grad_v_x*grad_u_x*(j_comp_out==1)*(k_comp_out==1) 
                                        //    + grad_v_y*grad_u_y*(j_comp_out==1)*(k_comp_out==1) );
                                    } else if (_st_op || _mix_le_op) {
                                        double p, q;
                                        if (_pressure_field_id==field_one || _pressure_field_id==field_two) {
                                            if (j_to_cell_i>=0) {
                                                q = _pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, qn+1);
                                            } else {
                                                q = 0.0;
                                            }
                                            if (k_to_cell_i>=0) {
                                                p = _pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, qn+1);
                                            } else {
                                                p = 0.0;
                                            }
                                        }
                                        if (_velocity_field_id==field_one && _velocity_field_id==field_two) {
                                            if (_st_op) {
                                                double grad_v_dot_grad_u = 0.0;
                                                for (int d1=0; d1<_ndim_requested; ++d1) {
                                                    for (int d2=0; d2<_ndim_requested; ++d2) {
                                                        grad_v_dot_grad_u += grad_v[d1]*(j_comp_out==d2)*grad_u[d1]*(k_comp_out==d2);
                                                    }
                                                }
                                                t_contribution += q_wt * _diffusion* grad_v_dot_grad_u;
                                                //(
                                                //      grad_v_x*grad_u_x*(j_comp_out==0)*(k_comp_out==0) 
                                                //    + grad_v_y*grad_u_y*(j_comp_out==0)*(k_comp_out==0) 
                                                //    + grad_v_x*grad_u_x*(j_comp_out==1)*(k_comp_out==1) 
                                                //    + grad_v_y*grad_u_y*(j_comp_out==1)*(k_comp_out==1) );
                                            } else if (_mix_le_op) {
                                                double div_v = 0.0, div_u = 0.0;
                                                for (int d=0; d<_ndim_requested; ++d) {
                                                    div_v += grad_v[d]*(j_comp_out==d);
                                                    div_u += grad_u[d]*(k_comp_out==d);
                                                }
                                                t_contribution += q_wt * -2./(scalar_type)(_ndim_requested) * _shear * div_v * div_u;
                                                //t_contribution += q_wt * ( -2./(scalar_type)(_ndim_requested) * _shear * (grad_v_x*(j_comp_out==0) + grad_v_y*(j_comp_out==1)) 
                                                //    * (grad_u_x*(k_comp_out==0) + grad_u_y*(k_comp_out==1)) );
                                       
                                                double sigma_v_dot_sigma_u = 0.0;
                                                for (int d1=0; d1<_ndim_requested; ++d1) {
                                                    for (int d2=0; d2<_ndim_requested; ++d2) {
                                                        sigma_v_dot_sigma_u += (0.5*grad_v[d1]*(j_comp_out==d2)+0.5*grad_v[d2]*(j_comp_out==d1))*(0.5*grad_u[d1]*(k_comp_out==d2)+0.5*grad_u[d2]*(k_comp_out==d1));
                                                    }
                                                }
                                                t_contribution += q_wt * 2 * _shear * sigma_v_dot_sigma_u;
                                                //t_contribution += q_wt * 2 * _shear* (
                                                //      grad_v_x*grad_u_x*(j_comp_out==0)*(k_comp_out==0) 
                                                //    + grad_v_y*grad_u_y*(j_comp_out==1)*(k_comp_out==1) 
                                                //    + 0.5*(grad_v_y*(j_comp_out==0) + grad_v_x*(j_comp_out==1))*(grad_u_y*(k_comp_out==0) + grad_u_x*(k_comp_out==1)));

                                            }
                                        } else if (_velocity_field_id==field_one && _pressure_field_id==field_two) {
                                            double div_v = 0.0;
                                            for (int d=0; d<_ndim_requested; ++d) {
                                                div_v += grad_v[d]*(j_comp_out==d);
                                            }
                                            t_contribution -= q_wt * div_v * p * pressure_coeff;
                                            //      (grad_v_x*(j_comp_out==0) + grad_v_y*(j_comp_out==1)) * p * pressure_coeff);
                                        } else if (_pressure_field_id==field_one && _velocity_field_id==field_two) {
                                            if (!_use_pinning || particle_j!=0) { // q is pinned at particle 0
                                                double div_u = 0.0;
                                                for (int d=0; d<_ndim_requested; ++d) {
                                                    div_u += grad_u[d]*(k_comp_out==d);
                                                }
                                                t_contribution -= q_wt * div_u * q * pressure_coeff;
                                                //      (grad_u_x*(k_comp_out==0) + grad_u_y*(k_comp_out==1)) * q * pressure_coeff);
                                            } 
                                            //else if ((particle_j==particle_k) && (particle_j==i)) {
                                            //    t_contribution += 1;
                                            //}
                                        } else {
                                            if (_mix_le_op) {
                                                if (!_use_pinning || particle_j!=0) { // q is pinned at particle 0
                                                    t_contribution -= q_wt * ( p * q * pressure_coeff );
                                                } 
                                            }
                                        }
                                        //else {
                                        //    if (particle_j==particle_k) {
                                        //        t_contribution += 1;
                                        //    }
                                        //}
                                    }
                                } 
                                else if (q_type==2) { // side on exterior

                                    if (_rd_op) {
                                        const double jumpv = v;
                                        const double jumpu = u;
                                        //const double avgv_x = grad_v_x;
                                        //const double avgv_y = grad_v_y;
                                        //const double avgu_x = grad_u_x;
                                        //const double avgu_y = grad_u_y;
                                        XYZ avg_grad_v;
                                        XYZ avg_grad_u;
                                        // unique outward normal
                                        XYZ n;
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            avg_grad_v[d] = grad_v[d];
                                            avg_grad_u[d] = grad_u[d];
                                            n[d] = unit_normals(i,_ndim_requested*qn+d);
                                        }

                                        //const XYZ n = XYZ(unit_normals(i,_ndim_requested*qn+0), unit_normals(i,_ndim_requested*qn+1), (_ndim_requested==3) ? unit_normals(i,_ndim_requested*qn+2) : 0);
                                        //auto jump_v_x = n_x*v;
                                        //auto jump_v_y = n_y*v;
                                        //auto jump_u_x = n_x*u;
                                        //auto jump_u_y = n_y*u;
                                        //auto jump_u_jump_v = jump_v_x*jump_u_x + jump_v_y*jump_u_y;

                                        t_contribution += penalty * q_wt * jumpv * jumpu;

                                        double avg_grad_v_dot_n = 0, avg_grad_u_dot_n = 0;
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            avg_grad_v_dot_n += avg_grad_v[d] * n[d];
                                            avg_grad_u_dot_n += avg_grad_u[d] * n[d];
                                        }
                                        t_contribution -= q_wt * _diffusion * avg_grad_v_dot_n * jumpu;
                                        t_contribution -= q_wt * _diffusion * avg_grad_u_dot_n * jumpv;
                                    } else {
                                        // only jump penalization currently included
                                        const double jumpv = v;
                                        const double jumpu = u;
                                        XYZ avg_grad_v;
                                        XYZ avg_grad_u;
                                        // unique outward normal
                                        XYZ n;
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            avg_grad_v[d] = grad_v[d];
                                            avg_grad_u[d] = grad_u[d];
                                            n[d] = unit_normals(i,_ndim_requested*qn+d);
                                        }
                                        //const double avgv_x = grad_v_x;
                                        //const double avgv_y = grad_v_y;
                                        //const double avgu_x = grad_u_x;
                                        //const double avgu_y = grad_u_y;

                                        //const auto n_x = unit_normals(i,2*qn+0); // unique outward normal
                                        //const auto n_y = unit_normals(i,2*qn+1);

                                        if (_le_op) {

                                            if (_use_vms) {
                                                const int current_edge_num_in_current_cell = (qn - num_interior_quadrature)/num_exterior_quadrature_per_side;
                                                Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> tau_edge(_ndim_requested, _ndim_requested);
                                                Teuchos :: SerialDenseSolver <local_index_type, scalar_type> vmsdg_solver;
                                                tau_edge.putScalar(0.0);
                                                for (int j1 = 0; j1 < _ndim_requested; ++j1) {
                                                    for (int j2 = 0; j2 < _ndim_requested; ++j2) {
                                                        tau_edge(j1, j2) = _tau(i, current_edge_num_in_current_cell, j1, j2);    //tau^(1)_s from 'current' elem
                                                    }
                                                }
                                                vmsdg_solver.setMatrix(Teuchos::rcp(&tau_edge, false));
                                                // Compute tau_edge (invert LHS matrix)
                                                auto info = vmsdg_solver.invert();
                                                double vmsBCfactor = 2.0;
                                                t_contribution += q_wt * jumpv * jumpu * vmsBCfactor * tau_edge(j_comp_out, k_comp_out); //MA: VMSDG term
                                            } else if (_use_sip) {
                                                t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out);
                                            }
                                            
                                            double mu_sigma_v_dot_n_dot_jump_u = 0.0;
                                            double mu_sigma_u_dot_n_dot_jump_v = 0.0;
                                            for (int d1=0; d1<_ndim_requested; ++d1) {
                                                for (int d2=0; d2<_ndim_requested; ++d2) {
                                                    mu_sigma_v_dot_n_dot_jump_u += 0.5*n[d1]*jumpu*(k_comp_out==d2)*(avg_grad_v[d1]*(j_comp_out==d2) + avg_grad_v[d2]*(j_comp_out==d1));
                                                    mu_sigma_u_dot_n_dot_jump_v += 0.5*n[d1]*jumpv*(j_comp_out==d2)*(avg_grad_u[d1]*(k_comp_out==d2) + avg_grad_u[d2]*(k_comp_out==d1));
                                                }
                                            }

                                            double n_dot_jump_v = 0.0, n_dot_jump_u = 0.0;
                                            for (int d=0; d<_ndim_requested; ++d) {
                                                n_dot_jump_v += n[d] * jumpv * (j_comp_out==d);
                                                n_dot_jump_u += n[d] * jumpu * (k_comp_out==d);
                                            }
                                            double lambda_sigma_v_dot_n_dot_jump_u = 0.0;
                                            double lambda_sigma_u_dot_n_dot_jump_v = 0.0;
                                            for (int d=0; d<_ndim_requested; ++d) {
                                                lambda_sigma_v_dot_n_dot_jump_u += avg_grad_v[d]*(j_comp_out==d) * n_dot_jump_u;
                                                lambda_sigma_u_dot_n_dot_jump_v += avg_grad_u[d]*(k_comp_out==d) * n_dot_jump_v;
                                            }
                                            t_contribution -= q_wt * 2 * _shear * mu_sigma_v_dot_n_dot_jump_u;
                                            t_contribution -= q_wt * 2 * _shear * mu_sigma_u_dot_n_dot_jump_v;
                                            t_contribution -= q_wt * _lambda * lambda_sigma_v_dot_n_dot_jump_u;
                                            t_contribution -= q_wt * _lambda * lambda_sigma_u_dot_n_dot_jump_v;
                                            //t_contribution -= q_wt * (
                                            //      2 * _shear * (n_x*avgv_x*(j_comp_out==0) 
                                            //        + 0.5*n_y*(avgv_y*(j_comp_out==0) + avgv_x*(j_comp_out==1))) * jumpu*(k_comp_out==0)  
                                            //    + 2 * _shear * (n_y*avgv_y*(j_comp_out==1) 
                                            //        + 0.5*n_x*(avgv_x*(j_comp_out==1) + avgv_y*(j_comp_out==0))) * jumpu*(k_comp_out==1)
                                            //    + _lambda*(avgv_x*(j_comp_out==0) + avgv_y*(j_comp_out==1))*(n_x*jumpu*(k_comp_out==0) + n_y*jumpu*(k_comp_out==1)));
                                            //t_contribution -= q_wt * (
                                            //      2 * _shear * (n_x*avgu_x*(k_comp_out==0) 
                                            //        + 0.5*n_y*(avgu_y*(k_comp_out==0) + avgu_x*(k_comp_out==1))) * jumpv*(j_comp_out==0)  
                                            //    + 2 * _shear * (n_y*avgu_y*(k_comp_out==1) 
                                            //        + 0.5*n_x*(avgu_x*(k_comp_out==1) + avgu_y*(k_comp_out==0))) * jumpv*(j_comp_out==1)
                                            //    + _lambda*(avgu_x*(k_comp_out==0) + avgu_y*(k_comp_out==1))*(n_x*jumpv*(j_comp_out==0) + n_y*jumpv*(j_comp_out==1)));
                                        } else if (_vl_op) {
                                            t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out);

                                            double sigma_v_dot_n_dot_jump_u = 0.0, sigma_u_dot_n_dot_jump_v = 0.0;
                                            for (int d1=0; d1<_ndim_requested; ++d1) {
                                                for (int d2=0; d2<_ndim_requested; ++d2) {
                                                    sigma_v_dot_n_dot_jump_u += n[d1]*avg_grad_v[d1]*(j_comp_out==d2) * jumpu*(k_comp_out==d2);
                                                    sigma_u_dot_n_dot_jump_v += n[d1]*avg_grad_u[d1]*(k_comp_out==d2) * jumpv*(j_comp_out==d2);
                                                }
                                            }
                                            t_contribution -= q_wt * _shear * sigma_v_dot_n_dot_jump_u;
                                            t_contribution -= q_wt * _shear * sigma_u_dot_n_dot_jump_v;

                                            //t_contribution -= q_wt * (
                                            //      _shear * (n_x*avgv_x*(j_comp_out==0) 
                                            //        + n_y*avgv_y*(j_comp_out==0)) * jumpu*(k_comp_out==0)  
                                            //    + _shear * (n_x*avgv_x*(j_comp_out==1) 
                                            //        + n_y*avgv_y*(j_comp_out==1)) * jumpu*(k_comp_out==1) );
                                            //t_contribution -= q_wt * (
                                            //      _shear * (n_x*avgu_x*(k_comp_out==0) 
                                            //        + n_y*avgu_y*(k_comp_out==0)) * jumpv*(j_comp_out==0)  
                                            //    + _shear * (n_x*avgu_x*(k_comp_out==1) 
                                            //        + n_y*avgu_y*(k_comp_out==1)) * jumpv*(j_comp_out==1) );

                                        } else if (_st_op || _mix_le_op) {
                                            double p, q;
                                            if (_pressure_field_id==field_one || _pressure_field_id==field_two) {
                                                if (j_to_cell_i>=0) {
                                                    q = _pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, qn+1);
                                                } else {
                                                    q = 0.0;
                                                }
                                                if (k_to_cell_i>=0) {
                                                    p = _pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, qn+1);
                                                } else {
                                                    p = 0.0;
                                                }
                                            }
                                            // p basis same as single component of u
                                            const double avgp = p;
                                            const double avgq = q;

                                            if (_velocity_field_id==field_one && _velocity_field_id==field_two) {
                                                if (_st_op) {
                                                    t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out);
                                                    double sigma_v_dot_n_dot_jump_u = 0.0, sigma_u_dot_n_dot_jump_v = 0.0;
                                                    for (int d1=0; d1<_ndim_requested; ++d1) {
                                                        for (int d2=0; d2<_ndim_requested; ++d2) {
                                                            sigma_v_dot_n_dot_jump_u += n[d1]*avg_grad_v[d1]*(j_comp_out==d2) * jumpu*(k_comp_out==d2);
                                                            sigma_u_dot_n_dot_jump_v += n[d1]*avg_grad_u[d1]*(k_comp_out==d2) * jumpv*(j_comp_out==d2);
                                                        }
                                                    }
                                                    t_contribution -= q_wt * _diffusion * sigma_v_dot_n_dot_jump_u;
                                                    t_contribution -= q_wt * _diffusion * sigma_u_dot_n_dot_jump_v;
                                                    //t_contribution -= q_wt * (
                                                    //      _diffusion * (n_x*avgv_x*(j_comp_out==0) 
                                                    //        + n_y*avgv_y*(j_comp_out==0)) * jumpu*(k_comp_out==0)  
                                                    //    + _diffusion * (n_x*avgv_x*(j_comp_out==1) 
                                                    //        + n_y*avgv_y*(j_comp_out==1)) * jumpu*(k_comp_out==1) );
                                                    //t_contribution -= q_wt * (
                                                    //      _diffusion * (n_x*avgu_x*(k_comp_out==0) 
                                                    //        + n_y*avgu_y*(k_comp_out==0)) * jumpv*(j_comp_out==0)  
                                                    //    + _diffusion * (n_x*avgu_x*(k_comp_out==1) 
                                                    //        + n_y*avgu_y*(k_comp_out==1)) * jumpv*(j_comp_out==1) );
                                                } else if (_mix_le_op) {
                                                    t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out);

                                                    double mu_sigma_v_dot_n_dot_jump_u = 0.0;
                                                    double mu_sigma_u_dot_n_dot_jump_v = 0.0;
                                                    for (int d1=0; d1<_ndim_requested; ++d1) {
                                                        for (int d2=0; d2<_ndim_requested; ++d2) {
                                                            mu_sigma_v_dot_n_dot_jump_u += 0.5*n[d1]*jumpu*(k_comp_out==d2)*(avg_grad_v[d1]*(j_comp_out==d2) + avg_grad_v[d2]*(j_comp_out==d1));
                                                            mu_sigma_u_dot_n_dot_jump_v += 0.5*n[d1]*jumpv*(j_comp_out==d2)*(avg_grad_u[d1]*(k_comp_out==d2) + avg_grad_u[d2]*(k_comp_out==d1));
                                                        }
                                                    }

                                                    t_contribution -= q_wt * 2 * _shear * mu_sigma_v_dot_n_dot_jump_u;
                                                    t_contribution -= q_wt * 2 * _shear * mu_sigma_u_dot_n_dot_jump_v;

                                                    double avg_div_v = 0, avg_div_u = 0;
                                                    for (int d=0; d<_ndim_requested; ++d) {
                                                        avg_div_v += avg_grad_v[d]*(j_comp_out==d);
                                                        avg_div_u += avg_grad_u[d]*(k_comp_out==d);
                                                    }
                                                    double n_dot_jump_v = 0.0, n_dot_jump_u = 0.0;
                                                    for (int d=0; d<_ndim_requested; ++d) {
                                                        n_dot_jump_v += n[d] * jumpv * (j_comp_out==d);
                                                        n_dot_jump_u += n[d] * jumpu * (k_comp_out==d);
                                                    }
                                                    t_contribution += q_wt * 2./(scalar_type)(_ndim_requested) * _shear * avg_div_v * n_dot_jump_u;
                                                    t_contribution += q_wt * 2./(scalar_type)(_ndim_requested) * _shear * avg_div_u * n_dot_jump_v;

                                                    //t_contribution -= q_wt * (
                                                    //      2 * _shear * (n_x*avgv_x*(j_comp_out==0) 
                                                    //        + 0.5*n_y*(avgv_y*(j_comp_out==0) + avgv_x*(j_comp_out==1))) * jumpu*(k_comp_out==0)  
                                                    //    + 2 * _shear * (n_y*avgv_y*(j_comp_out==1) 
                                                    //        + 0.5*n_x*(avgv_x*(j_comp_out==1) + avgv_y*(j_comp_out==0))) * jumpu*(k_comp_out==1)
                                                    //    - 2./(scalar_type)(_ndim_requested) * _shear * (avgv_x*(j_comp_out==0) + avgv_y*(j_comp_out==1))*(n_x*jumpu*(k_comp_out==0) + n_y*jumpu*(k_comp_out==1)));
                                                    //t_contribution -= q_wt * (
                                                    //      2 * _shear * (n_x*avgu_x*(k_comp_out==0) 
                                                    //        + 0.5*n_y*(avgu_y*(k_comp_out==0) + avgu_x*(k_comp_out==1))) * jumpv*(j_comp_out==0)  
                                                    //    + 2 * _shear * (n_y*avgu_y*(k_comp_out==1) 
                                                    //        + 0.5*n_x*(avgu_x*(k_comp_out==1) + avgu_y*(k_comp_out==0))) * jumpv*(j_comp_out==1)
                                                    //    - 2./(scalar_type)(_ndim_requested) * _shear * (avgu_x*(k_comp_out==0) + avgu_y*(k_comp_out==1))*(n_x*jumpv*(j_comp_out==0) + n_y*jumpv*(j_comp_out==1)));
                                                }
                                            } else if (_velocity_field_id==field_one && _pressure_field_id==field_two) {
                                                double n_dot_jump_v = 0.0;
                                                for (int d=0; d<_ndim_requested; ++d) {
                                                    n_dot_jump_v += n[d] * jumpv * (j_comp_out==d);
                                                }
                                                t_contribution += q_wt * pressure_coeff * avgp * n_dot_jump_v;
                                                //t_contribution += q_wt * n_dot_jump_v;
                                                //t_contribution += q_wt * (
                                                //      pressure_coeff * avgp * n_dot_jump_v );
                                            } else if (_pressure_field_id==field_one && _velocity_field_id==field_two) {
                                                if (!_use_pinning || particle_j!=0) { // q is pinned at particle 0
                                                    double n_dot_jump_u = 0.0;
                                                    for (int d=0; d<_ndim_requested; ++d) {
                                                        n_dot_jump_u += n[d] * jumpu * (k_comp_out==d);
                                                    }
                                                    t_contribution += q_wt * pressure_coeff * avgq * n_dot_jump_u;
                                                    //t_contribution += q_wt * (
                                                    //      pressure_coeff * avgq * (n_x*jumpu*(k_comp_out==0) + n_y*jumpu*(k_comp_out==1)) );
                                                }
                                            }
                                        }
                                    }
                                    
                                }
                                else if (q_type==0)  { // side on interior
                                    const int adjacent_cell_local_index_q = adjacent_cell_local_index[qn];
                                    //int current_side_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_side;
                                    //int adjacent_cell_local_index = (int)(adjacent_elements(i, current_side_num));

                                    //if (i>adjacent_cell_local_index) {
                                    //
                                    //////printf("size is %lu\n", _halo_id_view.extent(0));
                                    //int adjacent_cell_local_index = -1;
                                    ////// loop all of i's neighbors to get local index
                                    ////for (int l=0; l<_cell_particles_neighborhood->getNumNeighbors(i); ++l) {
                                    ////    auto this_neighbor_local_index = _cell_particles_neighborhood->getNeighbor(i,l);
                                    ////    scalar_type gid;
                                    ////    if (this_neighbor_local_index < nlocal) {
                                    ////        gid = static_cast<scalar_type>(_id_view(this_neighbor_local_index,0));
                                    ////    } else {
                                    ////        gid = static_cast<scalar_type>(_halo_id_view(this_neighbor_local_index-nlocal,0));
                                    ////    }
                                    ////    //printf("gid %lu\n", gid);
                                    ////    if (adjacent_cell_global_index==gid) {
                                    ////        adjacent_cell_local_index = this_neighbor_local_index;
                                    ////        break;
                                    ////    }
                                    ////}
                                    ////if (adjacent_cell_local_index < 0) {
                                    ////    std::cout << "looking for " << adjacent_cell_global_index << std::endl;
                                    ////    for (int l=0; l<_cell_particles_neighborhood->getNumNeighbors(i); ++l) {
                                    ////        auto this_neighbor_local_index = _cell_particles_neighborhood->getNeighbor(i,l);
                                    ////        scalar_type gid;
                                    ////        if (this_neighbor_local_index < nlocal) {
                                    ////            gid = static_cast<scalar_type>(_id_view(this_neighbor_local_index,0));
                                    ////        } else {
                                    ////            gid = static_cast<scalar_type>(_halo_id_view(this_neighbor_local_index-nlocal,0));
                                    ////        }
                                    ////        //printf("gid %lu\n", gid);
                                    ////        std::cout << "see gid " << gid << std::endl;
                                    ////    }
                                    ////    for (int l=0; l<_halo_id_view.extent(0); ++l) {
                                    ////        printf("gid on halo %d\n", _halo_id_view(l,0));
                                    ////    }
                                    ////}
                                    if (i <= adjacent_cell_local_index_q) return;
                                    
                                    if (j_to_adjacent_cell[qn]<0 && k_to_adjacent_cell[qn]<0 && j_to_cell_i<0 && k_to_cell_i<0) return;
                                    
                                    const auto normal_direction_correction = (i > adjacent_cell_local_index_q) ? 1 : -1; 
                                    XYZ n;
                                    for (int d=0; d<_ndim_requested; ++d) {
                                        n[d] = normal_direction_correction * unit_normals(i,_ndim_requested*qn+d);
                                    }
                                    // gives a unique normal that is always outward to the cell with the lower index

                                    // gets quadrature # on adjacent cell (enumerates quadrature on 
                                    // side_of_cell_i_to_adjacent_cell in reverse due to orientation)
                                    int adjacent_q = 0;
                                    if (_ndim_requested==2) {
                                        adjacent_q = num_interior_quadrature + side_of_cell_i_to_adjacent_cell[qn]*num_exterior_quadrature_per_side + (num_exterior_quadrature_per_side - ((qn-num_interior_quadrature)%num_exterior_quadrature_per_side) - 1);
                                    } else {
                                        for (int alt_qn=0; alt_qn<num_exterior_quadrature_per_side; ++alt_qn) {
                                            int potential_adjacent_q = num_interior_quadrature + side_of_cell_i_to_adjacent_cell[qn]*num_exterior_quadrature_per_side + alt_qn;
                                            bool all_same = true;
                                            for (int d=0; d<3; ++d) {
                                                double diff_d = quadrature_points(adjacent_cell_local_index_q, _ndim_requested*potential_adjacent_q+d) - quadrature_points(i,_ndim_requested*qn+d);
                                                if (diff_d*diff_d > 1e-14) {
                                                    all_same = false;
                                                    break;
                                                }
                                            }
                                            if (all_same) {
                                                adjacent_q = potential_adjacent_q;
                                                break;
                                            }        
                                        }
                                    }


                                    //int adjacent_q = num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_side + ((q-num_interior_quadrature)%num_exterior_quadrature_per_side);

                                    //// diagnostic that quadrature matches up between adjacent_cell_local_index & adjacent_q 
                                    //// along with i & q
                                    //auto my_x = quadrature_points(i, _ndim_requested*qn+0);
                                    //auto their_x = quadrature_points(adjacent_cell_local_index_q, _ndim_requested*adjacent_q+0);
                                    //if (std::abs(my_x-their_x)>1e-14) printf("xm: %f, t: %f, d: %.16f\n", my_x, their_x, my_x-their_x);
                                    //auto my_y = quadrature_points(i, _ndim_requested*qn+1);
                                    //auto their_y = quadrature_points(adjacent_cell_local_index_q, _ndim_requested*adjacent_q+1);
                                    //if (std::abs(my_y-their_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_y, their_y, my_y-their_y);
                                    //auto my_wt = quadrature_weights(i, q);
                                    //auto their_wt = quadrature_weights(adjacent_cell_local_index, adjacent_q);
                                    //if (std::abs(my_wt-their_wt)>1e-14) printf("wm: %f, t: %f, d: %.16f\n", my_wt, their_wt, my_wt-their_wt);
                                    //auto my_y = quadrature_points(i, 2*q+1);
                                    //auto their_y = quadrature_points(adjacent_cell_local_index, 2*adjacent_q+1);
                                    //if (std::abs(my_y-their_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_y, their_y, my_y-their_y);


                                    // diagnostics for quadrature points
                                    //printf("b: %f\n", _vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index, j_to_adjacent_cell[qn], adjacent_q+1));
                                    //auto my_j_index = _cell_particles_neighborhood->getNeighbor(i,j_to_cell_i);
                                    //auto their_j_index = _cell_particles_neighborhood->getNeighbor(adjacent_cell_local_index, j_to_adjacent_cell[qn]);
                                    //printf("xm: %d, t: %d\n", my_j_index, their_j_index);
                                    //auto my_q_x = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 0);
                                    //auto my_q_y = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 1);
                                    //auto their_q_x = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 0);
                                    //auto their_q_y = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 1);
                                    //if (std::abs(my_q_x-their_q_x)>1e-14) printf("xm: %f, t: %f, d: %.16f\n", my_q_x, their_q_x, my_q_x-their_q_x);
                                    //if (std::abs(my_q_y-their_q_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_q_y, their_q_y, my_q_y-their_q_y);

                                    //auto my_q_x = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 0);
                                    //auto my_q_y = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 1);
                                    ////auto their_q_index = _kokkos_quadrature_neighbor_lists_host(adjacent_cell_local_index, 1+k_to_adjacent_cell[qn]);
                                    //auto their_q_x = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 0);
                                    //auto their_q_y = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 1);
                                    //if (std::abs(my_q_x-their_q_x)>1e-14) printf("xm: %f, t: %f, d: %.16f\n", my_q_x, their_q_x, my_q_x-their_q_x);
                                    //if (std::abs(my_q_y-their_q_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_q_y, their_q_y, my_q_y-their_q_y);
                                    //
                                    //    XYZ avg_grad_v;
                                    //    XYZ avg_grad_u;
                                    //    // unique outward normal
                                    //    XYZ n;
                                    //    for (int d=0; d<_ndim_requested; ++d) {
                                    //        avg_grad_v[d] = grad_v[d];
                                    //        avg_grad_u[d] = grad_u[d];
                                    //        n[d] = unit_normals(i,_ndim_requested*qn+d);
                                    //    }
 
                                    double other_v, other_u;
                                    if (_use_vector_gmls) other_v = (j_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, adjacent_cell_local_index_q, j_comp_out, j_to_adjacent_cell[qn], j_comp_in, adjacent_q+1) : 0.0;
                                    else other_v = (j_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index_q, j_to_adjacent_cell[qn], adjacent_q+1) : 0.0;
                                    const double jumpv = normal_direction_correction*(v-other_v);

                                    if (_use_vector_gmls) other_u = (k_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, adjacent_cell_local_index_q, k_comp_out, k_to_adjacent_cell[qn], k_comp_in, adjacent_q+1) : 0.0;
                                    else other_u = (k_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index_q, k_to_adjacent_cell[qn], adjacent_q+1) : 0.0;
                                    const double jumpu = normal_direction_correction*(u-other_u);


                                    //double other_grad_v_x, other_grad_v_y, other_grad_u_x, other_grad_u_y;
                                    XYZ other_grad_v, other_grad_u;
                                    if (_use_vector_grad_gmls) {
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            other_grad_v[d] = (j_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, adjacent_cell_local_index_q, j_comp_out, d, j_to_adjacent_cell[qn], j_comp_in, adjacent_q+1) : 0.0;
                                            other_grad_u[d] = (k_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, adjacent_cell_local_index_q, k_comp_out, d, k_to_adjacent_cell[qn], k_comp_in, adjacent_q+1) : 0.0;
                                            //other_grad_v_y = (j_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, adjacent_cell_local_index_q, j_comp_out, 1, j_to_adjacent_cell[qn], j_comp_in, adjacent_q+1) : 0.0;
                                            //other_grad_u_x = (k_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, adjacent_cell_local_index_q, k_comp_out, 0, k_to_adjacent_cell[qn], k_comp_in, adjacent_q+1) : 0.0;
                                            //other_grad_u_y = (k_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, adjacent_cell_local_index_q, k_comp_out, 1, k_to_adjacent_cell[qn], k_comp_in, adjacent_q+1) : 0.0;
                                        }
                                    } else {
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            other_grad_v[d] = (j_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index_q, d, j_to_adjacent_cell[qn], adjacent_q+1) : 0.0;
                                            other_grad_u[d] = (k_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index_q, d, k_to_adjacent_cell[qn], adjacent_q+1) : 0.0;
                                        }
                                        //other_grad_v_y = (j_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index_q, 1, j_to_adjacent_cell[qn], adjacent_q+1) : 0.0;
                                        //other_grad_u_x = (k_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index_q, 0, k_to_adjacent_cell[qn], adjacent_q+1) : 0.0;
                                        //other_grad_u_y = (k_to_adjacent_cell[qn]>=0) ? _vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index_q, 1, k_to_adjacent_cell[qn], adjacent_q+1) : 0.0;

                                    }

                                    XYZ avg_grad_v, avg_grad_u;
                                    for (int d=0; d<_ndim_requested; ++d) {
                                        avg_grad_v[d] = 0.5*(grad_v[d] + other_grad_v[d]);
                                        avg_grad_u[d] = 0.5*(grad_u[d] + other_grad_u[d]);
                                    }
                                    //const double avgv_x = 0.5*(grad_v_x+other_grad_v_x);
                                    //const double avgv_y = 0.5*(grad_v_y+other_grad_v_y);
                                    //const double avgu_x = 0.5*(grad_u_x+other_grad_u_x);
                                    //const double avgu_y = 0.5*(grad_u_y+other_grad_u_y);

                                    if (_rd_op) {
                                        t_contribution += penalty * q_wt * jumpv * jumpu; // other half will be added by other cell

                                        double avg_grad_v_dot_n = 0, avg_grad_u_dot_n = 0;
                                        for (int d=0; d<_ndim_requested; ++d) {
                                            avg_grad_v_dot_n += avg_grad_v[d] * n[d];
                                            avg_grad_u_dot_n += avg_grad_u[d] * n[d];
                                        }
                                        t_contribution -= q_wt * _diffusion * avg_grad_v_dot_n * jumpu;
                                        t_contribution -= q_wt * _diffusion * avg_grad_u_dot_n * jumpv;
                                        //t_contribution -= q_wt * _diffusion * (avgv_x * n_x + avgv_y * n_y) * jumpu;
                                        //t_contribution -= q_wt * _diffusion * (avgu_x * n_x + avgu_y * n_y) * jumpv;
                                    } else {
                                        if (_le_op) {
                                            if (_use_vms) {
                                                const int current_edge_num_in_current_cell = (qn - num_interior_quadrature)/num_exterior_quadrature_per_side;
                                                const int current_edge_num_in_adjacent_cell = (adjacent_q - num_interior_quadrature)/num_exterior_quadrature_per_side;

                                                Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> tau_current_cell(_ndim_requested, _ndim_requested);
                                                Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> tau_adjacent_cell(_ndim_requested, _ndim_requested);
                                                Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> tau_edge(_ndim_requested, _ndim_requested);
                                                Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> delta_current_cell(_ndim_requested, _ndim_requested);
                                                Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> delta_adjacent_cell(_ndim_requested, _ndim_requested);
                                                Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> delta_edge(_ndim_requested, _ndim_requested);
                                                Teuchos :: SerialDenseSolver <local_index_type, scalar_type> vmsdg_solver;

                                                tau_current_cell.putScalar(0.0);
                                                tau_adjacent_cell.putScalar(0.0);
                                                tau_edge.putScalar(0.0);
                                                delta_current_cell.putScalar(0.0);
                                                delta_adjacent_cell.putScalar(0.0);
                                                delta_edge.putScalar(0.0);
                                                for (int j1 = 0; j1 < _ndim_requested; ++j1) {
                                                    for (int j2 = 0; j2 < _ndim_requested; ++j2) {
                                                        tau_current_cell(j1, j2) = _tau(i, current_edge_num_in_current_cell, j1, j2);                                    //tau^(1)_s from 'current' elem
                                                        tau_adjacent_cell(j1, j2) = _tau( (int)adjacent_cell_local_index_q, current_edge_num_in_adjacent_cell, j1, j2);  //tau^(2)_s from 'adjacent' elem
                                                        tau_edge(j1, j2) = tau_current_cell(j1, j2) + tau_adjacent_cell(j1, j2);  //tau_edge is the inverse of this (inverted in-place below)
                                                    }
                                                }
                                                vmsdg_solver.setMatrix(Teuchos::rcp(&tau_edge, false));
                                                auto info = vmsdg_solver.invert();
                                                delta_current_cell.multiply( Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, tau_edge, tau_current_cell, 0.0 );
                                                delta_adjacent_cell.multiply( Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, tau_edge, tau_adjacent_cell, 0.0 );
                                                delta_edge.multiply( Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, tau_current_cell, delta_adjacent_cell, 0.0 );

                                                const double traction_u_current_x = n[0] * ( _lambda * ((k_comp_out==0)*grad_u[0] + (k_comp_out==1)*grad_u[1]) + 2.0*_shear * (k_comp_out==0)*grad_u[0] ) +
                                                                                    n[1] * _shear * ((k_comp_out==0)*grad_u[1] + (k_comp_out==1)*grad_u[0]);
                                                const double traction_u_current_y = n[1] * ( _lambda * ((k_comp_out==0)*grad_u[0] + (k_comp_out==1)*grad_u[1]) + 2.0*_shear * (k_comp_out==1)*grad_u[1] ) +
                                                                                    n[0] * _shear * ((k_comp_out==0)*grad_u[1] + (k_comp_out==1)*grad_u[0]);
                                                const double traction_u_adjacent_x = n[0] * ( _lambda * ((k_comp_out==0)*other_grad_u[0] + (k_comp_out==1)*other_grad_u[1]) + 2.0*_shear * (k_comp_out==0)*other_grad_u[0] ) +
                                                                                     n[1] * _shear * ((k_comp_out==0)*other_grad_u[1] + (k_comp_out==1)*other_grad_u[0]);
                                                const double traction_u_adjacent_y = n[1] * ( _lambda * ((k_comp_out==0)*other_grad_u[0] + (k_comp_out==1)*other_grad_u[1]) + 2.0*_shear * (k_comp_out==1)*other_grad_u[1] ) +
                                                                                     n[0] * _shear * ((k_comp_out==0)*other_grad_u[1] + (k_comp_out==1)*other_grad_u[0]);
                                                const double traction_v_current_x = n[0] * ( _lambda * ((j_comp_out==0)*grad_v[0] + (j_comp_out==1)*grad_v[1]) + 2.0*_shear * (j_comp_out==0)*grad_v[0] ) +
                                                                                    n[1] * _shear * ((j_comp_out==0)*grad_v[1] + (j_comp_out==1)*grad_v[0]);
                                                const double traction_v_current_y = n[1] * ( _lambda * ((j_comp_out==0)*grad_v[0] + (j_comp_out==1)*grad_v[1]) + 2.0*_shear * (j_comp_out==1)*grad_v[1] ) +
                                                                                    n[0] * _shear * ((j_comp_out==0)*grad_v[1] + (j_comp_out==1)*grad_v[0]);
                                                const double traction_v_adjacent_x = n[0] * ( _lambda * ((j_comp_out==0)*other_grad_v[0] + (j_comp_out==1)*other_grad_v[1]) + 2.0*_shear * (j_comp_out==0)*other_grad_v[0] ) +
                                                                                     n[1] * _shear * ((j_comp_out==0)*other_grad_v[1] + (j_comp_out==1)*other_grad_v[0]);
                                                const double traction_v_adjacent_y = n[1] * ( _lambda * ((j_comp_out==0)*other_grad_v[0] + (j_comp_out==1)*other_grad_v[1]) + 2.0*_shear * (j_comp_out==1)*other_grad_v[1] ) +
                                                                                     n[0] * _shear * ((j_comp_out==0)*other_grad_v[1] + (j_comp_out==1)*other_grad_v[0]);
                        
                                                const double avg_traction_u_x = delta_current_cell(0, 0) * traction_u_current_x +
                                                                                delta_current_cell(0, 1) * traction_u_current_y +
                                                                                delta_adjacent_cell(0, 0) * traction_u_adjacent_x +
                                                                                delta_adjacent_cell(0, 1) * traction_u_adjacent_y;

                                                const double avg_traction_u_y = delta_current_cell(1, 0) * traction_u_current_x +
                                                                                delta_current_cell(1, 1) * traction_u_current_y +
                                                                                delta_adjacent_cell(1, 0) * traction_u_adjacent_x +
                                                                                delta_adjacent_cell(1, 1) * traction_u_adjacent_y;
  
                                                const double avg_traction_v_x = delta_current_cell(0, 0) * traction_v_current_x +
                                                                                delta_current_cell(0, 1) * traction_v_current_y +
                                                                                delta_adjacent_cell(0, 0) * traction_v_adjacent_x +
                                                                                delta_adjacent_cell(0, 1) * traction_v_adjacent_y;

                                                const double avg_traction_v_y = delta_current_cell(1, 0) * traction_v_current_x +
                                                                                delta_current_cell(1, 1) * traction_v_current_y +
                                                                                delta_adjacent_cell(1, 0) * traction_v_adjacent_x +
                                                                                delta_adjacent_cell(1, 1) * traction_v_adjacent_y;
  
                                                const double jump_traction_u_x = traction_u_current_x - traction_u_adjacent_x;
                                                const double jump_traction_u_y = traction_u_current_y - traction_u_adjacent_y;
                                                const double jump_traction_v_x = traction_v_current_x - traction_v_adjacent_x;
                                                const double jump_traction_v_y = traction_v_current_y - traction_v_adjacent_y;

                                                t_contribution += q_wt * jumpv * jumpu * tau_edge(j_comp_out, k_comp_out); 
                                                                      
                                                t_contribution -= q_wt * (
                                                                        jumpv * (j_comp_out==0) * avg_traction_u_x +
                                                                        jumpv * (j_comp_out==1) * avg_traction_u_y);
                                                                     
                                                t_contribution -= q_wt * (
                                                                        jumpu * (k_comp_out==0) * avg_traction_v_x +
                                                                        jumpu * (k_comp_out==1) * avg_traction_v_y);
                                                                     
                                                t_contribution -= q_wt * (
                                                                        jump_traction_v_x * ( delta_edge(0, 0) * jump_traction_u_x + delta_edge(0, 1) * jump_traction_u_y ) +
                                                                        jump_traction_v_y * ( delta_edge(1, 0) * jump_traction_u_x + delta_edge(1, 1) * jump_traction_u_y ) );
                                            } else if (_use_sip) {
                                                t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out);

                                                double mu_sigma_v_dot_n_dot_jump_u = 0.0;
                                                double mu_sigma_u_dot_n_dot_jump_v = 0.0;
                                                for (int d1=0; d1<_ndim_requested; ++d1) {
                                                    for (int d2=0; d2<_ndim_requested; ++d2) {
                                                        mu_sigma_v_dot_n_dot_jump_u += 0.5*n[d1]*jumpu*(k_comp_out==d2)*(avg_grad_v[d1]*(j_comp_out==d2) + avg_grad_v[d2]*(j_comp_out==d1));
                                                        mu_sigma_u_dot_n_dot_jump_v += 0.5*n[d1]*jumpv*(j_comp_out==d2)*(avg_grad_u[d1]*(k_comp_out==d2) + avg_grad_u[d2]*(k_comp_out==d1));
                                                    }
                                                }

                                                double n_dot_jump_v = 0.0, n_dot_jump_u = 0.0;
                                                for (int d=0; d<_ndim_requested; ++d) {
                                                    n_dot_jump_v += n[d] * jumpv * (j_comp_out==d);
                                                    n_dot_jump_u += n[d] * jumpu * (k_comp_out==d);
                                                }
                                                double lambda_sigma_v_dot_n_dot_jump_u = 0.0;
                                                double lambda_sigma_u_dot_n_dot_jump_v = 0.0;
                                                for (int d=0; d<_ndim_requested; ++d) {
                                                    lambda_sigma_v_dot_n_dot_jump_u += avg_grad_v[d]*(j_comp_out==d) * n_dot_jump_u;
                                                    lambda_sigma_u_dot_n_dot_jump_v += avg_grad_u[d]*(k_comp_out==d) * n_dot_jump_v;
                                                }
                                                t_contribution -= q_wt * 2 * _shear * mu_sigma_v_dot_n_dot_jump_u;
                                                t_contribution -= q_wt * 2 * _shear * mu_sigma_u_dot_n_dot_jump_v;
                                                t_contribution -= q_wt * _lambda * lambda_sigma_v_dot_n_dot_jump_u;
                                                t_contribution -= q_wt * _lambda * lambda_sigma_u_dot_n_dot_jump_v;
                                                //t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out); // other half will be added by other cell
                                                //t_contribution -= q_wt * (
                                                //      2 * _shear * (n_x*avgv_x*(j_comp_out==0) 
                                                //        + 0.5*n_y*(avgv_y*(j_comp_out==0) + avgv_x*(j_comp_out==1))) * jumpu*(k_comp_out==0)  
                                                //    + 2 * _shear * (n_y*avgv_y*(j_comp_out==1) 
                                                //        + 0.5*n_x*(avgv_x*(j_comp_out==1) + avgv_y*(j_comp_out==0))) * jumpu*(k_comp_out==1)
                                                //    + _lambda*(avgv_x*(j_comp_out==0) + avgv_y*(j_comp_out==1))*(n_x*jumpu*(k_comp_out==0) + n_y*jumpu*(k_comp_out==1)));
                                                //t_contribution -= q_wt * (
                                                //      2 * _shear * (n_x*avgu_x*(k_comp_out==0) 
                                                //        + 0.5*n_y*(avgu_y*(k_comp_out==0) + avgu_x*(k_comp_out==1))) * jumpv*(j_comp_out==0)  
                                                //    + 2 * _shear * (n_y*avgu_y*(k_comp_out==1) 
                                                //        + 0.5*n_x*(avgu_x*(k_comp_out==1) + avgu_y*(k_comp_out==0))) * jumpv*(j_comp_out==1)
                                                //    + _lambda*(avgu_x*(k_comp_out==0) + avgu_y*(k_comp_out==1))*(n_x*jumpv*(j_comp_out==0) + n_y*jumpv*(j_comp_out==1)));
                                            }
                                        } else if (_vl_op) {
                                            t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out);

                                            double sigma_v_dot_n_dot_jump_u = 0.0, sigma_u_dot_n_dot_jump_v = 0.0;
                                            for (int d1=0; d1<_ndim_requested; ++d1) {
                                                for (int d2=0; d2<_ndim_requested; ++d2) {
                                                    sigma_v_dot_n_dot_jump_u += n[d1]*avg_grad_v[d1]*(j_comp_out==d2) * jumpu*(k_comp_out==d2);
                                                    sigma_u_dot_n_dot_jump_v += n[d1]*avg_grad_u[d1]*(k_comp_out==d2) * jumpv*(j_comp_out==d2);
                                                }
                                            }
                                            t_contribution -= q_wt * _shear * sigma_v_dot_n_dot_jump_u;
                                            t_contribution -= q_wt * _shear * sigma_u_dot_n_dot_jump_v;
                                            //t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out); // other half will be added by other cell
                                            //t_contribution -= q_wt * (
                                            //      _shear * (n_x*avgv_x*(j_comp_out==0) 
                                            //        + n_y*avgv_y*(j_comp_out==0)) * jumpu*(k_comp_out==0)  
                                            //    + _shear * (n_x*avgv_x*(j_comp_out==1) 
                                            //        + n_y*avgv_y*(j_comp_out==1)) * jumpu*(k_comp_out==1) );
                                            //t_contribution -= q_wt * (
                                            //      _shear * (n_x*avgu_x*(k_comp_out==0) 
                                            //        + n_y*avgu_y*(k_comp_out==0)) * jumpv*(j_comp_out==0)  
                                            //    + _shear * (n_x*avgu_x*(k_comp_out==1) 
                                            //        + n_y*avgu_y*(k_comp_out==1)) * jumpv*(j_comp_out==1) );
                                        } else if (_st_op || _mix_le_op) {
                                            double p, q;
                                            if (_pressure_field_id==field_one || _pressure_field_id==field_two) {
                                                if (j_to_cell_i>=0) {
                                                    q = _pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, qn+1);
                                                } else {
                                                    q = 0.0;
                                                }
                                                if (k_to_cell_i>=0) {
                                                    p = _pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, qn+1);
                                                } else {
                                                    p = 0.0;
                                                }
                                            }
                                            const double other_q = (j_to_adjacent_cell[qn]>=0) ? _pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index_q, j_to_adjacent_cell[qn], adjacent_q+1) : 0.0;
                                            const double other_p = (k_to_adjacent_cell[qn]>=0) ? _pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index_q, k_to_adjacent_cell[qn], adjacent_q+1) : 0.0;
                                            const double avgp = 0.5*(p+other_p);
                                            const double avgq = 0.5*(q+other_q);

                                            if (_velocity_field_id==field_one && _velocity_field_id==field_two) {
                                                if (_st_op) {
                                                    t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out);
                                                    double sigma_v_dot_n_dot_jump_u = 0.0, sigma_u_dot_n_dot_jump_v = 0.0;
                                                    for (int d1=0; d1<_ndim_requested; ++d1) {
                                                        for (int d2=0; d2<_ndim_requested; ++d2) {
                                                            sigma_v_dot_n_dot_jump_u += n[d1]*avg_grad_v[d1]*(j_comp_out==d2) * jumpu*(k_comp_out==d2);
                                                            sigma_u_dot_n_dot_jump_v += n[d1]*avg_grad_u[d1]*(k_comp_out==d2) * jumpv*(j_comp_out==d2);
                                                        }
                                                    }
                                                    t_contribution -= q_wt * _diffusion * sigma_v_dot_n_dot_jump_u;
                                                    t_contribution -= q_wt * _diffusion * sigma_u_dot_n_dot_jump_v;
                                                    //t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out); // other half will be added by other cell
                                                    //t_contribution -= q_wt * (
                                                    //      _diffusion * (n_x*avgv_x*(j_comp_out==0) 
                                                    //        + n_y*avgv_y*(j_comp_out==0)) * jumpu*(k_comp_out==0)  
                                                    //    + _diffusion * (n_x*avgv_x*(j_comp_out==1) 
                                                    //        + n_y*avgv_y*(j_comp_out==1)) * jumpu*(k_comp_out==1) );
                                                    //t_contribution -= q_wt * (
                                                    //      _diffusion * (n_x*avgu_x*(k_comp_out==0) 
                                                    //        + n_y*avgu_y*(k_comp_out==0)) * jumpv*(j_comp_out==0)  
                                                    //    + _diffusion * (n_x*avgu_x*(k_comp_out==1) 
                                                    //        + n_y*avgu_y*(k_comp_out==1)) * jumpv*(j_comp_out==1) );
                                                } else if (_mix_le_op) {
                                                    t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out);

                                                    double mu_sigma_v_dot_n_dot_jump_u = 0.0;
                                                    double mu_sigma_u_dot_n_dot_jump_v = 0.0;
                                                    for (int d1=0; d1<_ndim_requested; ++d1) {
                                                        for (int d2=0; d2<_ndim_requested; ++d2) {
                                                            mu_sigma_v_dot_n_dot_jump_u += 0.5*n[d1]*jumpu*(k_comp_out==d2)*(avg_grad_v[d1]*(j_comp_out==d2) + avg_grad_v[d2]*(j_comp_out==d1));
                                                            mu_sigma_u_dot_n_dot_jump_v += 0.5*n[d1]*jumpv*(j_comp_out==d2)*(avg_grad_u[d1]*(k_comp_out==d2) + avg_grad_u[d2]*(k_comp_out==d1));
                                                        }
                                                    }

                                                    t_contribution -= q_wt * 2 * _shear * mu_sigma_v_dot_n_dot_jump_u;
                                                    t_contribution -= q_wt * 2 * _shear * mu_sigma_u_dot_n_dot_jump_v;

                                                    double avg_div_v = 0, avg_div_u = 0;
                                                    for (int d=0; d<_ndim_requested; ++d) {
                                                        avg_div_v += avg_grad_v[d]*(j_comp_out==d);
                                                        avg_div_u += avg_grad_u[d]*(k_comp_out==d);
                                                    }
                                                    double n_dot_jump_v = 0.0, n_dot_jump_u = 0.0;
                                                    for (int d=0; d<_ndim_requested; ++d) {
                                                        n_dot_jump_v += n[d] * jumpv * (j_comp_out==d);
                                                        n_dot_jump_u += n[d] * jumpu * (k_comp_out==d);
                                                    }
                                                    t_contribution += q_wt * 2./(scalar_type)(_ndim_requested) * _shear * avg_div_v * n_dot_jump_u;
                                                    t_contribution += q_wt * 2./(scalar_type)(_ndim_requested) * _shear * avg_div_u * n_dot_jump_v;
                                                    //t_contribution += penalty * q_wt * jumpv*jumpu*(j_comp_out==k_comp_out); // other half will be added by other cell
                                                    //t_contribution -= q_wt * (
                                                    //      2 * _shear * (n_x*avgv_x*(j_comp_out==0) 
                                                    //        + 0.5*n_y*(avgv_y*(j_comp_out==0) + avgv_x*(j_comp_out==1))) * jumpu*(k_comp_out==0)  
                                                    //    + 2 * _shear * (n_y*avgv_y*(j_comp_out==1) 
                                                    //        + 0.5*n_x*(avgv_x*(j_comp_out==1) + avgv_y*(j_comp_out==0))) * jumpu*(k_comp_out==1)
                                                    //    - 2./(scalar_type)(_ndim_requested) * _shear * (avgv_x*(j_comp_out==0) + avgv_y*(j_comp_out==1))*(n_x*jumpu*(k_comp_out==0) + n_y*jumpu*(k_comp_out==1)));
                                                    //t_contribution -= q_wt * (
                                                    //      2 * _shear * (n_x*avgu_x*(k_comp_out==0) 
                                                    //        + 0.5*n_y*(avgu_y*(k_comp_out==0) + avgu_x*(k_comp_out==1))) * jumpv*(j_comp_out==0)  
                                                    //    + 2 * _shear * (n_y*avgu_y*(k_comp_out==1) 
                                                    //        + 0.5*n_x*(avgu_x*(k_comp_out==1) + avgu_y*(k_comp_out==0))) * jumpv*(j_comp_out==1)
                                                    //    - 2./(scalar_type)(_ndim_requested) * _shear * (avgu_x*(k_comp_out==0) + avgu_y*(k_comp_out==1))*(n_x*jumpv*(j_comp_out==0) + n_y*jumpv*(j_comp_out==1)));
                                                }
                                            } else if (_velocity_field_id==field_one && _pressure_field_id==field_two) {
                                                double n_dot_jump_v = 0.0;
                                                for (int d=0; d<_ndim_requested; ++d) {
                                                    n_dot_jump_v += n[d] * jumpv * (j_comp_out==d);
                                                }
                                                t_contribution += q_wt * pressure_coeff * avgp * n_dot_jump_v;
                                                //t_contribution += q_wt * (
                                                //      pressure_coeff * avgp * (n_x*jumpv*(j_comp_out==0) + n_y*jumpv*(j_comp_out==1)) );
                                            } else if (_pressure_field_id==field_one && _velocity_field_id==field_two) {
                                                if (!_use_pinning || particle_j!=0) { // q is pinned at particle 0
                                                    double n_dot_jump_u = 0.0;
                                                    for (int d=0; d<_ndim_requested; ++d) {
                                                        n_dot_jump_u += n[d] * jumpu * (k_comp_out==d);
                                                    }
                                                    t_contribution += q_wt * pressure_coeff * avgq * n_dot_jump_u;
                                                    //t_contribution += q_wt * (
                                                    //      pressure_coeff * avgq * (n_x*jumpu*(k_comp_out==0) + n_y*jumpu*(k_comp_out==1)) );
                                                }
                                            }
                                        }
                                    }
                                }
                                compadre_assert_debug(t_contribution==t_contribution && "NaN encountered in t_contribution.");
                            //}
                            }, contribution);
                            teamMember.team_barrier();
                        }

                        val_data[0] = contribution;
                        if (j_has_value && k_has_value) {
                            this->_A->sumIntoLocalValues(row, 1, &val_data[0], &col_data[0], true);//, /*atomics*/false);
                        }
                    }
                    }
                }
            }
            }
        }
    //}
    //area = t_area;
    //perimeter = t_perimeter;
    //}, Kokkos::Sum<scalar_type>(area));
    });
    Kokkos::fence();
        if ((_st_op || _mix_le_op) && _use_pinning) {
            // this is the pin for pressure
            if ((field_one==field_two) && (field_one==_pressure_field_id)) {
                local_index_type row = local_to_dof_map(0, field_one, 0);
                double val_data[1] = {1.0};
                int    col_data[1] = {row};
                this->_A->sumIntoLocalValues(row, 1, &val_data[0], &col_data[0], true);
            }
        }
    } else if (field_one == _lagrange_field_id && field_two == _pressure_field_id) {
        // row all DOFs for solution against Lagrange Multiplier
        Teuchos::Array<local_index_type> col_data(nlocal * fields[field_two]->nDim());
        Teuchos::Array<scalar_type> val_data(nlocal * fields[field_two]->nDim());
        Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
        Teuchos::ArrayView<scalar_type> vals = Teuchos::ArrayView<scalar_type>(val_data);
    
        for (local_index_type l = 0; l < nlocal; l++) {
            for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                cols[l*fields[field_two]->nDim() + n] = local_to_dof_map(l, field_two, n);
                vals[l*fields[field_two]->nDim() + n] = 1;
            }
        }
        auto comm = this->_particles->getCoordsConst()->getComm();
        if (comm->getRank() == 0) {
            // local index 0 is the shared global id for the lagrange multiplier
            this->_A->sumIntoLocalValues(0, cols, vals);
        } else {
            // local index 0 is the shared global id for the lagrange multiplier
            this->_A->sumIntoLocalValues(nlocal, cols, vals);
        }
    } else if (field_one == _pressure_field_id && field_two == _lagrange_field_id) {
    
        // col all DOFs for solution against Lagrange Multiplier
        auto comm = this->_particles->getCoordsConst()->getComm();
        //if (comm->getRank() == 0) {
            for(local_index_type i = 0; i < nlocal; i++) {
                for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
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
        //}
    } else if (field_one == _lagrange_field_id && field_two == _lagrange_field_id) {
        // identity on all DOFs for Lagrange Multiplier (even DOFs not really used, since we only use first LM dof for now)
        scalar_type eps_penalty = nlocal; // would be nglobal but problem is serial
        //scalar_type eps_penalty = 1e-5;
    
        auto comm = this->_particles->getCoordsConst()->getComm();
        for(local_index_type i = 0; i < nlocal; i++) {
            for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
                local_index_type row = local_to_dof_map(i, field_one, k);
    
                Teuchos::Array<local_index_type> col_data(fields[field_two]->nDim());
                Teuchos::Array<scalar_type> val_data(fields[field_two]->nDim());
                Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
                Teuchos::ArrayView<scalar_type> vals = Teuchos::ArrayView<scalar_type>(val_data);
                for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                    if (i==0 && comm->getRank()==0) {
                        cols[n] = local_to_dof_map(i, field_two, n);
                        vals[n] = eps_penalty;
                    } else {
                        cols[n] = local_to_dof_map(i, field_two, n);
                        vals[n] = 1;
                    }
                }
                //#pragma omp critical
                {
                    this->_A->sumIntoLocalValues(row, cols, vals);
                }
            }
        }
    }


    // DIAGNOSTIC:: get global area
    //scalar_type global_area;
    //Teuchos::Ptr<scalar_type> global_area_ptr(&global_area);
    //Teuchos::reduceAll<int, scalar_type>(*(_cells->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, area, global_area_ptr);
    //if (_cells->getCoordsConst()->getComm()->getRank()==0) {
    //    printf("GLOBAL AREA: %.16f\n", global_area);
    //}
    //scalar_type global_perimeter;
    //Teuchos::Ptr<scalar_type> global_perimeter_ptr(&global_perimeter);
    //Teuchos::reduceAll<int, scalar_type>(*(_cells->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, perimeter, global_perimeter_ptr);
    //if (_cells->getCoordsConst()->getComm()->getRank()==0) {
    //    printf("GLOBAL PERIMETER: %.16f\n", global_perimeter);///((double)(num_interior_sides)));
    //}

    TEUCHOS_ASSERT(!this->_A.is_null());
    ComputeMatrixTime->stop();

}

const std::vector<InteractingFields> ReactionDiffusionPhysics::gatherFieldInteractions() {
    std::vector<InteractingFields> field_interactions;
    field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _velocity_field_id ));
    if (_st_op || _mix_le_op) {
        field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _velocity_field_id, _pressure_field_id));
        field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _pressure_field_id, _velocity_field_id));
        field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _pressure_field_id));
        if (_use_lm) {
            field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _lagrange_field_id, _pressure_field_id));
            field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _pressure_field_id, _lagrange_field_id));
            field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _lagrange_field_id));
        }
    }
    return field_interactions;
}
    
}
