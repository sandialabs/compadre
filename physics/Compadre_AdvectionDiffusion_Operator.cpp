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
 
    local_index_type ndim_requested = _parameters->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions");

    //
    // double-sized search
    //
    // it is a safe upper bound to use twice the previous search size as the maximum new radii
    auto max_h = _cell_particles_neighborhood->computeMaxHSupportSize(false /* local processor max is fine, since they are all the same */);
    scalar_type double_radius = 3.0 * max_h;
    scalar_type max_search_size = _cells->getCoordsConst()->getHaloSize();
    // check that max_halo_size is not violated by distance of the double jump
    const local_index_type comm_size = _cell_particles_neighborhood->getSourceCoordinates()->getComm()->getSize();
    if (comm_size > 1) {
        TEUCHOS_TEST_FOR_EXCEPT_MSG((max_search_size < double_radius), "Neighbor of neighbor search results in a search radius exceeding the halo size.");
    }

    auto neighbors_needed = GMLS::getNP(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 2);
    _particles_double_hop_neighborhood = Teuchos::rcp_static_cast<neighborhood_type>(Teuchos::rcp(
            new neighborhood_type(_cells, _particles.getRawPtr(), false /*material coords*/, maxLeaf)));
    _particles_double_hop_neighborhood->constructAllNeighborLists(max_search_size,
        "radius",
        true /*dry run for sizes*/,
        neighbors_needed+1,
        0.0, /* cutoff multiplier */
        double_radius, /* search size */
        false, /* uniform radii is true, but will be enforce automatically because all search sizes are the same globally */
        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));

    //
    // neighbor of neighbor search
    // 

    //_particles_double_hop_neighborhood = Teuchos::rcp(new MultiJumpNeighborhood(_cell_particles_neighborhood.getRawPtr()));
    //auto as_multi_jump = Teuchos::rcp_static_cast<MultiJumpNeighborhood>(_particles_double_hop_neighborhood);
    //as_multi_jump->constructNeighborOfNeighborLists(_cells->getCoordsConst()->getHaloSize());

    // generates id_view and halo_id_view which are entries that adjacent_elements refers to
	//_ids = Teuchos::rcp(new mvec_local_index_type(_cells->getCoordsConst()->getMapConst(), 1));
	//_id_view = _ids->getLocalView<Compadre::host_view_local_index_type>();
    //auto coords_map = _cells->getCoordsConst()->getMapConst(false); // map for locally owned coordinates
    //for (size_t i=0; i<_id_view.extent(0); ++i) {
    //    if (coords_map->getGlobalElement(i) > INT_MAX ) throw std::overflow_error("ID too large for int.");
    //    _id_view(i,0) = static_cast<int>(coords_map->getGlobalElement(i));
    //}
	//auto halo_importer = _cells->getCoordsConst()->getHaloImporterConst();
	//_halo_ids = Teuchos::rcp(new mvec_local_index_type(_cells->getCoordsConst()->getMapConst(true /*halo*/), 1, true /*setToZero*/));
	//_halo_ids->doImport(*_ids, *halo_importer, Tpetra::CombineMode::INSERT);
	//_halo_id_view = _halo_ids->getLocalView<Compadre::host_view_local_index_type>();


    //printf("owned size is %lu on p: %d\n", _id_view.extent(0), _cells->getCoordsConst()->getComm()->getRank());
    //printf("halo size is %lu on p: %d\n", _halo_id_view.extent(0), _cells->getCoordsConst()->getComm()->getRank());


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
        if (ndim_requested>2) _kokkos_augmented_source_coordinates_host(i,2) = coordinate.z;
    });

    Kokkos::View<double**> kokkos_target_coordinates("target_coordinates", target_coords->nLocal(), target_coords->nDim());
    _kokkos_target_coordinates_host = Kokkos::create_mirror_view(kokkos_target_coordinates);
    // fill in the target, adding regular coordiantes only into a kokkos view
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        xyz_type coordinate = target_coords->getLocalCoords(i, false /*include halo*/, use_physical_coords);
        _kokkos_target_coordinates_host(i,0) = coordinate.x;
        _kokkos_target_coordinates_host(i,1) = coordinate.y;
        if (ndim_requested>2) _kokkos_target_coordinates_host(i,2) = coordinate.z;
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
    
    // this quantity refers to global elements, so it's local number can be found by getting neighbor with same global number
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
    _gmls->addTargets(TargetOperation::GradientOfScalarPointEvaluation);
    _gmls->setAdditionalEvaluationSitesData(_kokkos_quadrature_neighbor_lists_host, _kokkos_quadrature_coordinates_host);
    _gmls->generateAlphas();



    // halo
 
    auto nlocal = target_coords->nLocal();
    auto nhalo = target_coords->nLocal(true)-nlocal;

    if (nhalo > 0) {
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
        for(local_index_type i = 0; i < nlocal; i++) {
            local_index_type num_neighbors = _cell_particles_neighborhood->getNumNeighbors(i);
            for (local_index_type l = 0; l < num_neighbors; l++) {
                if (_cell_particles_neighborhood->getNeighbor(i,l) >= nlocal) {
                   auto index = _cell_particles_neighborhood->getNeighbor(i,l)-nlocal;
                   halo_needed(index,0) = 1;
                }
            }
        }
        Kokkos::fence();

        _halo_big_to_small = host_view_local_index_type("halo big to small map", nhalo, 1);
        Kokkos::deep_copy(_halo_big_to_small, halo_needed);

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
            xyz_type coordinate = target_coords->getLocalCoords(nlocal+_halo_small_to_big(i,0), true /*include halo*/, use_physical_coords);
            kokkos_halo_target_coordinates_host(i,0) = coordinate.x;
            kokkos_halo_target_coordinates_host(i,1) = coordinate.y;
            if (ndim_requested>2) kokkos_halo_target_coordinates_host(i,2) = 0;//coordinate.z;
        });

        // make temporary particle set of halo target coordinates needed
		Teuchos::RCP<Compadre::ParticlesT> halo_particles =
			Teuchos::rcp( new Compadre::ParticlesT(_parameters, _cells->getCoordsConst()->getComm(), _parameters->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions")));
        halo_particles->resize(max_needed_entries, true);
        auto halo_coords = halo_particles->getCoords();
		//std::vector<Compadre::XyzVector> verts_to_insert(max_needed_entries);
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,max_needed_entries), KOKKOS_LAMBDA(const int i) {
            XyzVector coord_i(0,0,0);
            for (local_index_type j=0; j<ndim_requested; ++j) {
                coord_i[j] = kokkos_halo_target_coordinates_host(i,j);
            }
            halo_coords->replaceLocalCoords(i, coord_i, true);
            halo_coords->replaceLocalCoords(i, coord_i, false);
            //for (local_index_type j=0; j<ndim_requested; ++j) {
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
 
        // perform neighbor search to halo particles using same h as previous radii search
        _halo_cell_particles_neighborhood = Teuchos::rcp_static_cast<neighborhood_type>(Teuchos::rcp(
                new neighborhood_type(_cells, halo_particles.getRawPtr(), false /*material coords*/, maxLeaf)));
        _halo_cell_particles_neighborhood->constructAllNeighborLists(max_search_size /* we know we have enough halo info */,
            "radius",
            true /*dry run for sizes*/,
            neighbors_needed+1,
            0.0, /* cutoff multiplier */
            max_h, /* search size */
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
                kokkos_halo_quadrature_coordinates_host(i*_weights_ndim + j, 0) = 
                    halo_quadrature_points(_halo_small_to_big(i,0),2*j+0);
                kokkos_halo_quadrature_coordinates_host(i*_weights_ndim + j, 1) = 
                    halo_quadrature_points(_halo_small_to_big(i,0),2*j+1);
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
        _halo_gmls = Teuchos::rcp<GMLS>(new GMLS(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                     2 /* dimension */,
                     _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"), 
                     "STANDARD", "NO_CONSTRAINT"));
        _halo_gmls->setProblemData(kokkos_halo_neighbor_lists_host,
                        _kokkos_augmented_source_coordinates_host,
                        kokkos_halo_target_coordinates_host,
                        kokkos_halo_epsilons_host);
        _halo_gmls->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
        _halo_gmls->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));

        _halo_gmls->addTargets(TargetOperation::ScalarPointEvaluation);
        _halo_gmls->addTargets(TargetOperation::GradientOfScalarPointEvaluation);
        _halo_gmls->setAdditionalEvaluationSitesData(kokkos_halo_quadrature_neighbor_lists_host, kokkos_halo_quadrature_coordinates_host);
        _halo_gmls->generateAlphas();

    }

    GenerateData->stop();

}

local_index_type AdvectionDiffusionPhysics::getMaxNumNeighbors() {
    // _particles_double_hop_neighborhood set to null after computeGraph is called
    TEUCHOS_ASSERT(!_particles_double_hop_neighborhood.is_null());
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
    // set neighborhood to null because it is large (storage) and not used again
	//_particles_double_hop_neighborhood = Teuchos::null;
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

    auto halo_quadrature_points = _cells->getFieldManager()->getFieldByName("quadrature_points")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_quadrature_weights = _cells->getFieldManager()->getFieldByName("quadrature_weights")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_quadrature_type = _cells->getFieldManager()->getFieldByName("interior")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_unit_normals = _cells->getFieldManager()->getFieldByName("unit_normal")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_adjacent_elements = _cells->getFieldManager()->getFieldByName("adjacent_elements")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();


    auto penalty = (_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")+1)*_parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_cell_particles_neighborhood->computeMaxHSupportSize(true /* global processor max */);
    printf("Computed penalty parameter: %.16f\n", penalty); 
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
    //                } else { //                    //this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
    //                    this->_A->replaceLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
    //                    //rhs_vals(row,0) = contribution;
    //                    row_seen(row) = 1;
    //                }
    //                //this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
    //            }
    //        }
    //    }
    //}
    auto local_map = _cells->getCoordsConst()->getMapConst(false);
    auto halo_map = _cells->getCoordsConst()->getMapConst(true);

    double area = 0; // (good, no rewriting)
    double perimeter = 0;
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
    double t_perimeter = 0; 
 

    //printf("interior: %d, exterior: %d\n", num_interior_quadrature, num_exterior_quadrature_per_edge);
    int num_interior_edges = 0;
    // loop over cells including halo cells 
    for (int i=0; i<_cells->getCoordsConst()->nLocal(true); ++i) {

        // filter out (skip) all halo cells that are not able to be seen from a locally owned particle
        auto halo_i = (i<nlocal) ? -1 : _halo_big_to_small(i-nlocal,0);
        if (i>=nlocal /* halo */  && halo_i<0 /* not one found within max_h of locally owned particles */) continue;

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

        std::vector<double> edge_lengths(3);
        {
            int current_edge_num = 0;
            for (int q=0; q<_weights_ndim; ++q) {
                if (quadrature_type(i,q)!=1) { // edge
                    int new_current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                    if (new_current_edge_num!=current_edge_num) {
                        edge_lengths[new_current_edge_num] = quadrature_weights(i,q);
                        current_edge_num = new_current_edge_num;
                    } else {
                        edge_lengths[current_edge_num] += quadrature_weights(i,q);
                    }
                }
            }
        }

        //std::vector<int> adjacent_cells(3,-1); // initialize all entries to -1
        //int this_edge_num = -1;
        //for (int q=0; q<_weights_ndim; ++q) {
        //    int new_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
        //    if (new_edge_num != this_edge_num) {
        //        this_edge_num = new_edge_num;
        //        adjacent_cells[this_edge_num] = (int)(adjacent_elements(i, this_edge_num));
        //    }
        //}
        //TEUCHOS_ASSERT(this_edge_num==2);

        int this_edge_num = -1;
        for (int q=0; q<_weights_ndim; ++q) {
            this_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
            //printf("q: %d, e: %d\n", q, this_edge_num);
            if (i<nlocal) { // locally owned
                if (quadrature_type(i,q)==1) { // interior
                    //t_area += quadrature_weights(i,q);
                    auto x=quadrature_points(i,2*q+0);
                    auto y=quadrature_points(i,2*q+1);
                    t_area += quadrature_weights(i,q) * (2 + 3*x + 3*y);
                } else if (quadrature_type(i,q)==2) { // exterior quad point on exterior edge
                    //t_perimeter += quadrature_weights(i,q);
                    //t_perimeter += quadrature_weights(i,q ) * (quadrature_points(i,2*q+0)*quadrature_points(i,2*q+0) + quadrature_points(i,2*q+1)*quadrature_points(i,2*q+1));
                    // integral of x^2 + y^2 over (0,0),(2,0),(2,2),(0,2) is
                    // 4*8/3+2*8 = 26.666666
                    auto x=quadrature_points(i,2*q+0);
                    auto y=quadrature_points(i,2*q+1);
                    auto nx=unit_normals(i,2*q+0);
                    auto ny=unit_normals(i,2*q+1);
                    t_perimeter += quadrature_weights(i,q) * (x+x*x+x*y+y+y*y) * (nx + ny);
                } else if (quadrature_type(i,q)==0) { // interior quad point on exterior edge
                    //if (i>adjacent_elements(i,this_edge_num)) {
                    //    t_perimeter += quadrature_weights(i,q ) * 1 / edge_lengths[this_edge_num];
                    //    if (q%3==0) num_interior_edges++;
                    //}
                    //auto x=quadrature_points(i,2*q+0);
                    //auto y=quadrature_points(i,2*q+1);
                    //auto nx=unit_normals(i,2*q+0);
                    //auto ny=unit_normals(i,2*q+1);
                    int adjacent_cell = adjacent_elements(i,this_edge_num);
                    int side_i_to_adjacent_cell = -1;
                    for (int z=0; z<3; ++z) {
                        if ((int)adjacent_elements(adjacent_cell,z)==i) {
                            side_i_to_adjacent_cell = z;
                        }
                    }
                    
                    int adjacent_q = num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge + (num_exterior_quadrature_per_edge - ((q-num_interior_quadrature)%num_exterior_quadrature_per_edge) - 1);

                    //auto x=quadrature_points(adjacent_cell, num_interior_quadrature + 2*(side_i_to_adjacent_cell+(q%3))+0);
                    //auto y=quadrature_points(adjacent_cell, num_interior_quadrature + 2*(side_i_to_adjacent_cell+(q%3))+1);
                    //auto x=quadrature_points(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge+((q-num_interior_quadrature)%num_exterior_quadrature_per_edge))+0);
                    //auto y=quadrature_points(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge+((q-num_interior_quadrature)%num_exterior_quadrature_per_edge))+1);
                    auto x=quadrature_points(adjacent_cell, 2*adjacent_q+0);
                    auto y=quadrature_points(adjacent_cell, 2*adjacent_q+1);

                    //auto y=quadrature_points(adjacent_cell,2*q+1);
                    //
                    auto nx=-unit_normals(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge+((q-num_interior_quadrature)%num_exterior_quadrature_per_edge))+0);
                    auto ny=-unit_normals(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge+((q-num_interior_quadrature)%num_exterior_quadrature_per_edge))+1);
                    //auto nx=unit_normals(i,2*q+0);
                    //auto ny=unit_normals(i,2*q+1);
                    t_perimeter += quadrature_weights(i,q) * (x+x*x+x*y+y+y*y) * (nx + ny);
                }
            }
        }
        // get all particle neighbors of cell i
        //local_index_type num_neighbors = (i<nlocal) ? _cell_particles_neighborhood->getNumNeighbors(i)
        //    : _halo_cell_particles_neighborhood->getNumNeighbors(halo_i);

        local_index_type num_neighbors = (i<nlocal) ? _particles_double_hop_neighborhood->getNumNeighbors(i)
            : _halo_cell_particles_neighborhood->getNumNeighbors(halo_i);

        for (local_index_type j=0; j<num_neighbors; ++j) {
            auto particle_j = (i<nlocal) ? _particles_double_hop_neighborhood->getNeighbor(i,j)
                : _halo_cell_particles_neighborhood->getNeighbor(halo_i,j);
            //auto particle_j = (i<nlocal) ? _cell_particles_neighborhood->getNeighbor(i,j)
            //    : _halo_cell_particles_neighborhood->getNeighbor(halo_i,j);
            // particle_j is an index from {0..nlocal+nhalo}
            if (particle_j>=nlocal /* particle is halo particle */) continue; // row should be locally owned

            // locally owned DOF
            local_index_type row = local_to_dof_map(particle_j, field_one, 0 /* component 0*/);
            for (local_index_type k=0; k<num_neighbors; ++k) {
                auto particle_k = (i<nlocal) ? _particles_double_hop_neighborhood->getNeighbor(i,k)
                    : _halo_cell_particles_neighborhood->getNeighbor(halo_i,k);
                //auto particle_k = (i<nlocal) ? _cell_particles_neighborhood->getNeighbor(i,k)
                //    : _halo_cell_particles_neighborhood->getNeighbor(halo_i,k);
                // particle_k is an index from {0..nlocal+nhalo}

                // do filter out (skip) all halo particles that are not seen by local DOF
                // wrong thing to do, because k could be double hop through cell i

                col_data(0) = local_to_dof_map(particle_k, field_two, 0 /* component */);

                int j_to_cell_i = -1;
                int k_to_cell_i = -1;
                for (int l=0; l<_cell_particles_neighborhood->getNumNeighbors(i); ++l) {
                    auto this_neighbor = _cell_particles_neighborhood->getNeighbor(i,l);
                    if (this_neighbor==particle_j) {
                        j_to_cell_i = l;
                        if (k_to_cell_i >= 0) break;
                    }
                    if (this_neighbor==particle_k) {
                        k_to_cell_i = l;
                        if (j_to_cell_i >= 0) break;
                    }
                }

                //if (j_to_cell_i>=0) {// && k_to_cell_i>=0) {
                bool j_has_value = false;
                bool k_has_value = false;

                double contribution = 0;
                if (_parameters->get<Teuchos::ParameterList>("physics").get<bool>("l2 projection only")) {
                   for (int q=0; q<_weights_ndim; ++q) {
                       auto q_type = (i<nlocal) ? quadrature_type(i,q) : halo_quadrature_type(_halo_small_to_big(halo_i,0),q);
                       if (q_type==1) { // interior
                           auto q_wt = (i<nlocal) ? quadrature_weights(i,q) : halo_quadrature_weights(_halo_small_to_big(halo_i,0),q);
                           auto u = (i<nlocal) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1)
                                : _halo_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, k, q+1);
                           auto v = (i<nlocal) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1)
                                : _halo_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, q+1);

                           contribution += q_wt * u * v;
                       } 
	                   TEUCHOS_ASSERT(contribution==contribution);
                   }
                } else {
                    TEUCHOS_ASSERT(_cells->getCoordsConst()->getComm()->getSize()==1);
                    // loop over quadrature
                    //std::vector<double> edge_lengths(3);
                    //int current_edge_num = 0;
                    //for (int q=0; q<_weights_ndim; ++q) {
                    //    if (quadrature_type(i,q)!=1) { // edge
                    //        int new_current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                    //        if (new_current_edge_num!=current_edge_num) {
                    //            edge_lengths[new_current_edge_num] = quadrature_weights(i,q);
                    //            current_edge_num = new_current_edge_num;
                    //        } else {
                    //            edge_lengths[current_edge_num] += quadrature_weights(i,q);
                    //        }
                    //    }
                    //}
                    //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_kokkos_epsilons_host(i);
                    for (int q=0; q<_weights_ndim; ++q) {
                        double u, v;
                        double grad_u_x, grad_u_y, grad_v_x, grad_v_y;
                        
                        //printf("jto: %d, kto: %d\n", j_to_cell_i, k_to_cell_i);

                        if (j_to_cell_i>=0) {
                            v = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, q+1);
                            //v = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1);
                            grad_v_x = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, j_to_cell_i, q+1);
                            grad_v_y = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, j_to_cell_i, q+1);
                            j_has_value = true;
                        } else {
                            v = 0.0;
                            grad_v_x = 0.0;
                            grad_v_y = 0.0;
                        }
                        if (k_to_cell_i>=0) {
                            u = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, q+1);
                            //u = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1);
                            grad_u_x = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, k_to_cell_i, q+1);
                            grad_u_y = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, k_to_cell_i, q+1);
                            k_has_value = true;
                        } else {
                            u = 0.0;
                            grad_u_x = 0.0;
                            grad_u_y = 0.0;
                        }

                        //printf("i: %d, j: %d, k: %d, q: %d, contribution: %f\n", i, j, k, q, contribution);
                        if (quadrature_type(i,q)==1) { // interior
                           auto q_wt = (i<nlocal) ? quadrature_weights(i,q) : halo_quadrature_weights(_halo_small_to_big(halo_i,0),q);
                           //auto u = (i<nlocal) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1)
                           //     : _halo_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, k, q+1);
                           //auto v = (i<nlocal) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1)
                           //     : _halo_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, q+1);

                           contribution += _advection * q_wt * u * v;
                           //printf("u: %f, v: %f\n", u, v);
                           //contribution += _diffusion * quadrature_weights(i,q) 
                           //    * grad_v_x * grad_u_x;//_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, j, q+1) 
                           //    //* _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, k, q+1);
                           //contribution += _diffusion * quadrature_weights(i,q) 
                           //    * grad_v_y * grad_u_y;//_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, j, q+1) 
                           //    //* _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, k, q+1);
                        } 
                        else if (quadrature_type(i,q)==2) { // edge on exterior
                            //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_kokkos_epsilons_host(i);
                            int current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                            //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/edge_lengths[current_edge_num];
                            int adjacent_cell = adjacent_elements(i,current_edge_num);
                            TEUCHOS_ASSERT(adjacent_cell < 0);
                            // penalties for edges of mass
                            double jumpv = v;//_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, q+1);
                            double jumpu = u;//_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, q+1);
                            double avgv_x = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, j_to_cell_i, q+1);
                            double avgv_y = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, j_to_cell_i, q+1);
                            double avgu_x = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, k_to_cell_i, q+1);
                            double avgu_y = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, k_to_cell_i, q+1);
                            auto n_x = unit_normals(i,2*q+0); // unique normal
                            auto n_y = unit_normals(i,2*q+1);
                            auto jump_v_x = n_x*v;//_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, q+1);
                            auto jump_v_y = n_y*v;//_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, q+1);
                            auto jump_u_x = n_x*u;//_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, q+1);
                            auto jump_u_y = n_y*u;//_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, q+1);
                            auto jump_u_jump_v = jump_v_x*jump_u_x + jump_v_y*jump_u_y;

                            contribution += penalty * quadrature_weights(i,q) * jump_u_jump_v;//jumpv * jumpu; // other half will be added by other cell
                            //contribution += penalty * quadrature_weights(i,q) * jumpv * jumpu;
                            //contribution -= quadrature_weights(i,q) * _diffusion * (avgv_x * n_x + avgv_y * n_y) * jumpu;
                            //contribution -= quadrature_weights(i,q) * _diffusion * (avgu_x * n_x + avgu_y * n_y) * jumpv;
                            //printf("exterior edge\n");
                            
                        }
                        else if (quadrature_type(i,q)==0)  { // edge on interior
                            //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_kokkos_epsilons_host(i);
                            //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_kokkos_epsilons_host(i);
                            //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty");///_kokkos_epsilons_host(i);
                            int current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                            //auto penalty = _parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/edge_lengths[current_edge_num];
                            

                            //scalar_type adjacent_cell_global_index = adjacent_elements(i, current_edge_num);
                            int adjacent_cell_local_index = (int)(adjacent_elements(i, current_edge_num));
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
                            TEUCHOS_ASSERT(adjacent_cell_local_index >= 0);
                            int j_to_adjacent_cell = -1;
                            int k_to_adjacent_cell = -1;
                            
                            TEUCHOS_ASSERT(adjacent_cell_local_index < nlocal); // must be local, for now
                            for (int l=0; l<_cell_particles_neighborhood->getNumNeighbors(adjacent_cell_local_index); ++l) {
                                auto this_neighbor = _cell_particles_neighborhood->getNeighbor(adjacent_cell_local_index,l);
                                if (this_neighbor==particle_j) {
                                    j_to_adjacent_cell = l;
                                    if (k_to_adjacent_cell >= 0) break;
                                } 
                                if (this_neighbor==particle_k) {
                                    k_to_adjacent_cell = l;
                                    if (j_to_adjacent_cell >= 0) break;
                                }
                            }
                            if (j_to_adjacent_cell>=0) {
                                j_has_value = true;
                            }
                            if (k_to_adjacent_cell>=0) {
                                k_has_value = true;
                            }
                            //if (j_to_adjacent_cell>=0 && k_to_adjacent_cell>=0) {
                            //    printf("both\n");
                            //} else if (j_to_adjacent_cell>=0 || k_to_adjacent_cell>=0) {
                            //    printf("one\n");
                            //} else {
                            //    printf("neither\n");
                            //}

                            ////printf("adjacent cell %d into %d\n", adjacent_cell,_cells->getCoordsConst()->nLocal());
                            //TEUCHOS_ASSERT(adjacent_cell_global_index >= 0);
                            //// penalties for edges of mass
                            ////
                            
                            // adjacent_cell_local_index >= nlocal implies target is off processor, however to avoid communication we
                            // recompute quantity

                            // also, j and k will NOT remain j and k when retrieved from adjacent_cell_local_index
                            // adjacent_cell_local_index may not see either j or k
                            
                            ////switch to only adding if normal_direction_correction is positive
                            ////then the half can be removed
                            ////check signs in jump (maybe backwards)
//                          //  auto normal_direction_correction = 1;//(i > adjacent_cell_local_index) ? -1 : 1; // gives a unique normal

                            auto normal_direction_correction = (i > adjacent_cell_local_index) ? -1 : 1; // gives a unique normal
                            //auto n_x = normal_direction_correction * unit_normals(i,2*q+0);
                            //auto n_y = normal_direction_correction * unit_normals(i,2*q+1);
                            auto n_x = unit_normals(i,2*q+0);
                            auto n_y = unit_normals(i,2*q+1);

                            double jump_u_jump_v = 0.0;
                            double avgu_x=0, avgv_x=0, avgu_y=0, avgv_y=0;
                            double integral_fraction = 1.0;
                            double jumpu = 0.0;
                            double jumpv = 0.0;

                            int side_i_to_adjacent_cell = -1;
                            for (int z=0; z<num_exterior_quadrature_per_edge; ++z) {
                                if ((int)(adjacent_elements(adjacent_cell_local_index,z))==i) {
                                    side_i_to_adjacent_cell = z;
                                }
                            }
                            int adjacent_q = num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge + (num_exterior_quadrature_per_edge - ((q-num_interior_quadrature)%num_exterior_quadrature_per_edge) - 1);
                            //int adjacent_q = num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge + ((q-num_interior_quadrature)%num_exterior_quadrature_per_edge);

                            // diagnostic that quadrature matches up between adjacent_cell_local_index & adjacent_q 
                            // along with i & q
                            auto my_x = quadrature_points(i, 2*q+0);
                            auto my_wt = quadrature_weights(i, q);
                            auto their_wt = quadrature_weights(adjacent_cell_local_index, adjacent_q);
                            if (std::abs(my_wt-their_wt)>1e-14) printf("wm: %f, t: %f, d: %.16f\n", my_wt, their_wt, my_wt-their_wt);
                            auto my_y = quadrature_points(i, 2*q+1);
                            auto their_x = quadrature_points(adjacent_cell_local_index, 2*adjacent_q+0);
                            auto their_y = quadrature_points(adjacent_cell_local_index, 2*adjacent_q+1);
                            if (std::abs(my_x-their_x)>1e-14) printf("xm: %f, t: %f, d: %.16f\n", my_x, their_x, my_x-their_x);
                            if (std::abs(my_y-their_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_y, their_y, my_y-their_y);

                            //if (j_to_adjacent_cell>=0 && k_to_adjacent_cell>=0) {// && i>adjacent_cell_local_index) {
                            //if (j_to_adjacent_cell>=0) {
                                //printf("a: %f\n", jumpv);
                                double other_v = (j_to_adjacent_cell>=0) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index, j_to_adjacent_cell, adjacent_q+1) : 0;
                                jumpv = v-other_v;//_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, q+1);
                                //printf("b: %f\n", _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index, j_to_adjacent_cell, adjacent_q+1));
                                auto my_j_index = _cell_particles_neighborhood->getNeighbor(i,j_to_cell_i);
                                auto their_j_index = _cell_particles_neighborhood->getNeighbor(adjacent_cell_local_index, j_to_adjacent_cell);
                                //printf("xm: %d, t: %d\n", my_j_index, their_j_index);
                                //auto my_q_x = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 0);
                                //auto my_q_y = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 1);
                                //auto their_q_x = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 0);
                                //auto their_q_y = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 1);
                                //if (std::abs(my_q_x-their_q_x)>1e-14) printf("xm: %f, t: %f, d: %.16f\n", my_q_x, their_q_x, my_q_x-their_q_x);
                                //if (std::abs(my_q_y-their_q_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_q_y, their_q_y, my_q_y-their_q_y);

                                auto my_q_x = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 0);
                                auto my_q_y = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 1);
                                //auto their_q_index = _kokkos_quadrature_neighbor_lists_host(adjacent_cell_local_index, 1+k_to_adjacent_cell);
                                auto their_q_x = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 0);
                                auto their_q_y = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 1);
                                if (std::abs(my_q_x-their_q_x)>1e-14) printf("xm: %f, t: %f, d: %.16f\n", my_q_x, their_q_x, my_q_x-their_q_x);
                                if (std::abs(my_q_y-their_q_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_q_y, their_q_y, my_q_y-their_q_y);
                            //} else {
                            //    jumpv = 0;
                            //}
                            //if (j_to_adjacent_cell>=0) {
                            //if (k_to_adjacent_cell>=0) {
                                double other_u = (k_to_adjacent_cell>=0) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index, k_to_adjacent_cell, adjacent_q+1) : 0.0;
                                //jumpu -= _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1);
                                jumpu = u-other_u;//_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1);
                                //if (k_to_adjacent_cell<0||k_to_cell_i) jumpu=0;
                                //if (j_to_adjacent_cell<0||j_to_cell_i) jumpv=0;
                            //} else {
                            //    jumpu = 0;
                            //}
                            //}


                            //if (j_to_adjacent_cell>=0 || k_to_adjacent_cell>=0) {
                            //    jumpv = normal_direction_correction * (_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1)
                            //        -_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index, j_to_adjacent_cell, q+1)*(j_to_adjacent_cell>=0));
                            //    jumpu = normal_direction_correction * (_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1)
                            //        -_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index, k_to_adjacent_cell, q+1)*(k_to_adjacent_cell>=0));


                            //    //j and k are local neighbor numbering on a cell i, not on cell adjacent_cell_local_index
                            //    //need to check that j and k are 

                            //    avgv_x = 0.5*(_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, j, q+1)
                            //                            + _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index, 0, j_to_adjacent_cell, q+1));
                            //    avgv_y = 0.5*(_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, j, q+1)
                            //                            + _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index, 1, j_to_adjacent_cell, q+1));
                            //    avgu_x = 0.5*(_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, k, q+1)
                            //                            + _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index, 0, k_to_adjacent_cell, q+1));
                            //    avgu_y = 0.5*(_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, k, q+1)
                            //                            + _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index, 1, k_to_adjacent_cell, q+1));

                            auto jump_v_x = n_x*v-n_x*other_v;
                            auto jump_v_y = n_y*v-n_y*other_v;
                            auto jump_u_x = n_x*u-n_x*other_u;
                            auto jump_u_y = n_y*u-n_y*other_u;
                            jump_u_jump_v = jump_v_x*jump_u_x + jump_v_y*jump_u_y;
                            //}
                            //if (j_to_adjacent_cell>=0 && k_to_adjacent_cell>=0) {
                            //    integral_fraction = 1.0;
                            //}

                            //contribution += 0.5* penalty * quadrature_weights(i,q) * jump_u_jump_v;//jumpv * jumpu; // other half will be added by other cell
                            contribution +=  0.5 * penalty * quadrature_weights(i,q) * jump_u_jump_v;//jumpv * jumpu; // other half will be added by other cell
                            //contribution += 0.5 * penalty * quadrature_weights(i,q) * jumpv * jumpu; // other half will be added by other cell
                            //contribution -= quadrature_weights(i,q) * _diffusion * (avgv_x * n_x + avgv_y * n_y) * jumpu;
                            //contribution -= quadrature_weights(i,q) * _diffusion * (avgu_x * n_x + avgu_y * n_y) * jumpv;
                            //contribution += penalty * 0.5 * quadrature_weights(i,q) * jumpv * jumpu; // other half will be added by other cell
                            //contribution -= 0.5 * quadrature_weights(i,q) * _diffusion * (avgv_x * n_x + avgv_y * n_y) * jumpu;
                            //contribution -= 0.5 * quadrature_weights(i,q) * _diffusion * (avgu_x * n_x + avgu_y * n_y) * jumpv;
                            //contribution += penalty * 0.5 * quadrature_weights(i,q) * jumpvr;// * jumpur; // other half will be added by other cell
                            // this cell numbering is only valid on serial (one) processor runs
                         //    TEUCHOS_ASSERT(adjacent_cell >= 0);
                            // penalties for edges of mass
                            //double jumpvr = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1)
                            //    -_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell, j, q+1);
                            //double jumpur = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k, q+1)
                            //    -_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell, k, q+1);
                            ////if (i<adjacent_cell) {
                            ////    contribution += penalty * quadrature_weights(i,q) * jumpvr * jumpur; // other half will be added by other cell
                            ////}
                            //contribution += penalty * 0.5 * quadrature_weights(i,q) * jumpvr * jumpur; // other half will be added by other cell
                            //printf("jumpvr: %f, jumpur: %f, penalty: %f, qw: %f\n", jumpvr, jumpur, penalty, quadrature_weights(i,q));
                            //printf("interior edge\n");
                            //}
                        }
                        //printf("i: %d, j: %d, k: %d, q: %d, contribution: %f\n", i, j, k, q, contribution);
                           TEUCHOS_ASSERT(contribution==contribution);
                    }
                    //for (int q=0; q<_weights_ndim; ++q) {
                    //    if (quadrature_type(i,q)==1) { // interior
                    //        contribution += quadrature_weights(i,q) 
                    //            * _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, j, q+1) 
                    //            * _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, k, q+1);
                    //        contribution += quadrature_weights(i,q) 
                    //            * _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, j, q+1) 
                    //            * _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, k, q+1);
                    //    } 
	                //    TEUCHOS_ASSERT(contribution==contribution);
                    //}
                }

                val_data(0) = contribution;
                if (j_has_value && k_has_value) {
                    this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
                }
                //}
            }
        }
    }
    area = t_area;
    perimeter = t_perimeter;
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
    // DIAGNOSTIC:: get global area
	scalar_type global_area;
	Teuchos::Ptr<scalar_type> global_area_ptr(&global_area);
	Teuchos::reduceAll<int, scalar_type>(*(_cells->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, area, global_area_ptr);
    if (_cells->getCoordsConst()->getComm()->getRank()==0) {
        printf("GLOBAL AREA: %.16f\n", global_area);
    }
	scalar_type global_perimeter;
	Teuchos::Ptr<scalar_type> global_perimeter_ptr(&global_perimeter);
	Teuchos::reduceAll<int, scalar_type>(*(_cells->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, perimeter, global_perimeter_ptr);
    if (_cells->getCoordsConst()->getComm()->getRank()==0) {
        printf("GLOBAL PERIMETER: %.16f\n", global_perimeter);///((double)(num_interior_edges)));
    }

	TEUCHOS_ASSERT(!this->_A.is_null());
	ComputeMatrixTime->stop();

}

const std::vector<InteractingFields> AdvectionDiffusionPhysics::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	return field_interactions;
}
    
}
