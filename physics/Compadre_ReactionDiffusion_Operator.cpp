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

    auto neighbors_needed = GMLS::getNP(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 2);

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
        _kokkos_epsilons_host(i) = epsilons(i,0);
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
            Kokkos::fence();
        }
    }

    GenerateData->stop();

}

local_index_type ReactionDiffusionPhysics::getMaxNumNeighbors() {
    // _particles_triple_hop_neighborhood set to null after computeGraph is called
    TEUCHOS_ASSERT(!_particles_triple_hop_neighborhood.is_null());
    return _particles_triple_hop_neighborhood->computeMaxNumNeighbors(false /*local processor maximum*/);
}

Teuchos::RCP<crs_graph_type> ReactionDiffusionPhysics::computeGraph(local_index_type field_one, local_index_type field_two) {
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
    // set neighborhood to null because it is large (storage) and not used again
	_particles_triple_hop_neighborhood = Teuchos::null;
	ComputeGraphTime->stop();
    return this->_A_graph;
}


void ReactionDiffusionPhysics::computeMatrix(local_index_type field_one, local_index_type field_two, scalar_type time) {

    bool use_physical_coords = true; // can be set on the operator in the future

    bool l2_op = (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="l2");
    bool rd_op = (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="rd");
    bool le_op = (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="le");

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

    double area = 0; // (good, no rewriting)
    double perimeter = 0;

    //double t_area = 0; 
    //double t_perimeter = 0; 
 

    // this maps should be for nLocal(true) so that it contains local neighbor lookups for owned and halo particles
    std::vector<std::map<local_index_type, local_index_type> > particle_to_local_neighbor_lookup(_cells->getCoordsConst()->nLocal(true), std::map<local_index_type, local_index_type>());
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_cells->getCoordsConst()->nLocal(true)), [&](const int i) {
    //for (int i=0; i<_cells->getCoordsConst()->nLocal(true); ++i) {
        std::map<local_index_type, local_index_type>* i_map = const_cast<std::map<local_index_type, local_index_type>* >(&particle_to_local_neighbor_lookup[i]);
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

    // loop over cells including halo cells 
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_cells->getCoordsConst()->nLocal(true)), [&](const int i, scalar_type& t_area) {
    //for (int i=0; i<_cells->getCoordsConst()->nLocal(true); ++i) {
        std::map<local_index_type, local_index_type> cell_neighbors;

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

        std::vector<double> edge_lengths(3);
        {
            int current_edge_num = 0;
            for (int q=0; q<_weights_ndim; ++q) {
                auto q_type = (i<nlocal) ? quadrature_type(i,q) 
                    : halo_quadrature_type(_halo_small_to_big(halo_i,0),q);
                if (q_type!=1) { // edge
                    auto q_wt = (i<nlocal) ? quadrature_weights(i,q) 
                        : halo_quadrature_weights(_halo_small_to_big(halo_i,0),q);
                    int new_current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                    if (new_current_edge_num!=current_edge_num) {
                        edge_lengths[new_current_edge_num] = q_wt;
                        current_edge_num = new_current_edge_num;
                    } else {
                        edge_lengths[current_edge_num] += q_wt;
                    }
                }
            }
        }

        // information needed for calculating jump terms
        std::vector<int> current_edge_num(_weights_ndim);
        std::vector<int> side_of_cell_i_to_adjacent_cell(_weights_ndim);
        std::vector<int> adjacent_cell_local_index(_weights_ndim);
        if (!l2_op) {
            TEUCHOS_ASSERT(_cells->getCoordsConst()->getComm()->getSize()==1);
            for (int q=0; q<_weights_ndim; ++q) {
                if (q>=num_interior_quadrature) {
                    current_edge_num[q] = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                    adjacent_cell_local_index[q] = (int)(adjacent_elements(i, current_edge_num[q]));
                    side_of_cell_i_to_adjacent_cell[q] = -1;
                    for (int z=0; z<num_exterior_quadrature_per_edge; ++z) {
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
            //        current_edge_num[q] = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
            //        adjacent_cell_local_index[q] = (i<nlocal) ? (int)(adjacent_elements(i, current_edge_num[q]))
            //            : (int)(halo_adjacent_elements(halo_i, current_edge_num[q]));
            //        side_of_cell_i_to_adjacent_cell[q] = -1;
            //        for (int z=0; z<num_exterior_quadrature_per_edge; ++z) {
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
//        //int this_edge_num = -1;
//        //for (int q=0; q<_weights_ndim; ++q) {
//        //    int new_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
//        //    if (new_edge_num != this_edge_num) {
//        //        this_edge_num = new_edge_num;
//        //        adjacent_cells[this_edge_num] = (int)(adjacent_elements(i, this_edge_num));
//        //    }
//        //}
//        //TEUCHOS_ASSERT(this_edge_num==2);
//
//        int this_edge_num = -1;
//        for (int q=0; q<_weights_ndim; ++q) {
//            this_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
//            //printf("q: %d, e: %d\n", q, this_edge_num);
//            if (i<nlocal) { // locally owned
//                if (quadrature_type(i,q)==1) { // interior
//                    //t_area += quadrature_weights(i,q);
//                    auto x=quadrature_points(i,2*q+0);
//                    auto y=quadrature_points(i,2*q+1);
//                    t_area += quadrature_weights(i,q) * (2 + 3*x + 3*y);
//                } else if (quadrature_type(i,q)==2) { // exterior quad point on exterior edge
//                    //t_perimeter += quadrature_weights(i,q);
//                    //t_perimeter += quadrature_weights(i,q ) * (quadrature_points(i,2*q+0)*quadrature_points(i,2*q+0) + quadrature_points(i,2*q+1)*quadrature_points(i,2*q+1));
//                    // integral of x^2 + y^2 over (0,0),(2,0),(2,2),(0,2) is
//                    // 4*8/3+2*8 = 26.666666
//                    auto x=quadrature_points(i,2*q+0);
//                    auto y=quadrature_points(i,2*q+1);
//                    auto nx=unit_normals(i,2*q+0);
//                    auto ny=unit_normals(i,2*q+1);
//                    t_perimeter += quadrature_weights(i,q) * (x+x*x+x*y+y+y*y) * (nx + ny);
//                } else if (quadrature_type(i,q)==0) { // interior quad point on exterior edge
//                    //if (i>adjacent_elements(i,this_edge_num)) {
//                    //    t_perimeter += quadrature_weights(i,q ) * 1 / edge_lengths[this_edge_num];
//                    //    if (q%3==0) num_interior_edges++;
//                    //}
//                    //auto x=quadrature_points(i,2*q+0);
//                    //auto y=quadrature_points(i,2*q+1);
//                    //auto nx=unit_normals(i,2*q+0);
//                    //auto ny=unit_normals(i,2*q+1);
//                    int adjacent_cell = adjacent_elements(i,this_edge_num);
//                    int side_i_to_adjacent_cell = -1;
//                    for (int z=0; z<3; ++z) {
//                        if ((int)adjacent_elements(adjacent_cell,z)==i) {
//                            side_i_to_adjacent_cell = z;
//                        }
//                    }
//                    
//                    int adjacent_q = num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge + (num_exterior_quadrature_per_edge - ((q-num_interior_quadrature)%num_exterior_quadrature_per_edge) - 1);
//
//                    //auto x=quadrature_points(adjacent_cell, num_interior_quadrature + 2*(side_i_to_adjacent_cell+(q%3))+0);
//                    //auto y=quadrature_points(adjacent_cell, num_interior_quadrature + 2*(side_i_to_adjacent_cell+(q%3))+1);
//                    //auto x=quadrature_points(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge+((q-num_interior_quadrature)%num_exterior_quadrature_per_edge))+0);
//                    //auto y=quadrature_points(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge+((q-num_interior_quadrature)%num_exterior_quadrature_per_edge))+1);
//                    auto x=quadrature_points(adjacent_cell, 2*adjacent_q+0);
//                    auto y=quadrature_points(adjacent_cell, 2*adjacent_q+1);
//
//                    //auto y=quadrature_points(adjacent_cell,2*q+1);
//                    //
//                    auto nx=-unit_normals(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge+((q-num_interior_quadrature)%num_exterior_quadrature_per_edge))+0);
//                    auto ny=-unit_normals(adjacent_cell, 2*(num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge+((q-num_interior_quadrature)%num_exterior_quadrature_per_edge))+1);
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
                cell_neighbors[particle_j] = 1;
            }
            if (!l2_op) {
                TEUCHOS_ASSERT(_cells->getCoordsConst()->getComm()->getSize()==1);
                for (local_index_type j=0; j<3; ++j) {
                    int adj_el = (int)(adjacent_elements(i,j));
                    if (adj_el>=0) {
                        num_neighbors = _cell_particles_neighborhood->getNumNeighbors(adj_el);
                        for (local_index_type k=0; k<num_neighbors; ++k) {
                            auto particle_k = _cell_particles_neighborhood->getNeighbor(adj_el,k);
                            cell_neighbors[particle_k] = 1;
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
            auto particle_j = j_it->first;
            //auto particle_j = (i<nlocal) ? _particles_double_hop_neighborhood->getNeighbor(i,j)
            //    : _halo_particles_double_hop_neighborhood->getNeighbor(halo_i,j);
            // particle_j is an index from {0..nlocal+nhalo}
            if (particle_j>=nlocal /* particle is halo particle */) continue; // row should be locally owned
            // locally owned DOF
            for (local_index_type j_comp = 0; j_comp < fields[field_one]->nDim(); ++j_comp) {

                local_index_type row = local_to_dof_map(particle_j, field_one, j_comp);

                int k=-1;
                //for (local_index_type k=0; k<num_neighbors; ++k) {
                for (auto k_it=cell_neighbors.begin(); k_it!=cell_neighbors.end(); ++k_it) {
                    k++;
                    auto particle_k = k_it->first;
                    //auto particle_k = (i<nlocal) ? _particles_double_hop_neighborhood->getNeighbor(i,k)
                    //    : _halo_particles_double_hop_neighborhood->getNeighbor(halo_i,k);
                    // particle_k is an index from {0..nlocal+nhalo}

                    // do filter out (skip) all halo particles that are not seen by local DOF
                    // wrong thing to do, because k could be double hop through cell i

                    for (local_index_type k_comp = 0; k_comp < fields[field_two]->nDim(); ++k_comp) {

                        col_data[0] = local_to_dof_map(particle_k, field_two, k_comp);

                        int j_to_cell_i = -1;
                        if (particle_to_local_neighbor_lookup[i].count(particle_j)==1){
                            j_to_cell_i = particle_to_local_neighbor_lookup[i][particle_j];
                        }
                        int k_to_cell_i = -1;
                        if (particle_to_local_neighbor_lookup[i].count(particle_k)==1){
                            k_to_cell_i = particle_to_local_neighbor_lookup[i][particle_k];
                        }

                        bool j_has_value = false;
                        bool k_has_value = false;

                        double contribution = 0;
                        if (l2_op) {
                            for (int q=0; q<_weights_ndim; ++q) {
                                auto q_type = (i<nlocal) ? quadrature_type(i,q) : halo_quadrature_type(_halo_small_to_big(halo_i,0),q);
                                if (q_type==1) { // interior
                                    double u, v;
                                    auto q_wt = (i<nlocal) ? quadrature_weights(i,q) : halo_quadrature_weights(_halo_small_to_big(halo_i,0),q);
                                    if (j_to_cell_i>=0) {
                                        v = (i<nlocal) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, q+1)
                                             : _halo_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j_to_cell_i, q+1);
                                        j_has_value = true;
                                    } else {
                                        v = 0;
                                    }
                                    if (k_to_cell_i>=0) {
                                        u = (i<nlocal) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, k_to_cell_i, q+1)
                                             : _halo_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, k_to_cell_i, q+1);
                                        k_has_value = true;
                                    } else {
                                        u = 0;
                                    }

                                    contribution += q_wt * u * v;
                                } 
	                            TEUCHOS_ASSERT(contribution==contribution);
                            }
                        } else {
                            TEUCHOS_ASSERT(_cells->getCoordsConst()->getComm()->getSize()==1);
                            // loop over quadrature
                            for (int q=0; q<_weights_ndim; ++q) {
                                auto q_type = quadrature_type(i,q);
                                auto q_wt = quadrature_weights(i,q);
                                double u, v;
                                double grad_u_x, grad_u_y, grad_v_x, grad_v_y;
                                
                                if (j_to_cell_i>=0) {
                                    v = _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j_to_cell_i, q+1);
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
                                    grad_u_x = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, k_to_cell_i, q+1);
                                    grad_u_y = _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, k_to_cell_i, q+1);
                                    k_has_value = true;
                                } else {
                                    u = 0.0;
                                    grad_u_x = 0.0;
                                    grad_u_y = 0.0;
                                }

                                if (q_type==1) { // interior

                                    if (rd_op) {
                                        contribution += _reaction * q_wt * u * v;
                                        contribution += _diffusion * q_wt 
                                            * grad_v_x * grad_u_x;
                                        contribution += _diffusion * q_wt 
                                            * grad_v_y * grad_u_y;
                                    } else if (le_op) {
                                        //if (particle_j==particle_k && particle_j==i && j_comp==k_comp) {
                                        //    contribution += _lambda*1./num_interior_quadrature;
                                        //}
                                        contribution += q_wt * _lambda * (grad_v_x*(j_comp==0) + grad_v_y*(j_comp==1)) 
                                            * (grad_u_x*(k_comp==0) + grad_u_y*(k_comp==1));
                                       
                                        contribution += q_wt * 2 * _shear* (
                                              grad_v_x*grad_u_x*(j_comp==0)*(k_comp==0) 
                                            + grad_v_y*grad_u_y*(j_comp==1)*(k_comp==1) 
                                            + 0.5*(grad_v_y*(j_comp==0) + grad_v_x*(j_comp==1))*(grad_u_y*(k_comp==0) + grad_u_x*(k_comp==1)));
                                    }
 
                                } 
                                else if (q_type==2) { // edge on exterior

                                    if (rd_op) {
                                        double jumpv = v;
                                        double jumpu = u;
                                        double avgv_x = grad_v_x;
                                        double avgv_y = grad_v_y;
                                        double avgu_x = grad_u_x;
                                        double avgu_y = grad_u_y;
                                        auto n_x = unit_normals(i,2*q+0); // unique outward normal
                                        auto n_y = unit_normals(i,2*q+1);
                                        //auto jump_v_x = n_x*v;
                                        //auto jump_v_y = n_y*v;
                                        //auto jump_u_x = n_x*u;
                                        //auto jump_u_y = n_y*u;
                                        //auto jump_u_jump_v = jump_v_x*jump_u_x + jump_v_y*jump_u_y;

                                        contribution += penalty * q_wt * jumpv * jumpu;
                                        contribution -= q_wt * _diffusion * (avgv_x * n_x + avgv_y * n_y) * jumpu;
                                        contribution -= q_wt * _diffusion * (avgu_x * n_x + avgu_y * n_y) * jumpv;
                                    } else if (le_op) {
                                        // only jump penalization currently included
                                        double jumpv = v;
                                        double jumpu = u;
                                        double avgv_x = grad_v_x;
                                        double avgv_y = grad_v_y;
                                        double avgu_x = grad_u_x;
                                        double avgu_y = grad_u_y;
                                        auto n_x = unit_normals(i,2*q+0); // unique outward normal
                                        auto n_y = unit_normals(i,2*q+1);

                                        contribution += penalty * q_wt * jumpv*jumpu*(j_comp==k_comp);
                                        contribution -= q_wt * (
                                              2 * _shear * (n_x*avgv_x*(j_comp==0) 
                                                + 0.5*n_y*(avgv_y*(j_comp==0) + avgv_x*(j_comp==1))) * jumpu*(k_comp==0)  
                                            + 2 * _shear * (n_y*avgv_y*(j_comp==1) 
                                                + 0.5*n_x*(avgv_x*(j_comp==1) + avgv_y*(j_comp==0))) * jumpu*(k_comp==1)
                                            + _lambda*(avgv_x*(j_comp==0) + avgv_y*(j_comp==1))*(n_x*jumpu*(k_comp==0) + n_y*jumpu*(k_comp==1)));
                                        contribution -= q_wt * (
                                              2 * _shear * (n_x*avgu_x*(k_comp==0) 
                                                + 0.5*n_y*(avgu_y*(k_comp==0) + avgu_x*(k_comp==1))) * jumpv*(j_comp==0)  
                                            + 2 * _shear * (n_y*avgu_y*(k_comp==1) 
                                                + 0.5*n_x*(avgu_x*(k_comp==1) + avgu_y*(k_comp==0))) * jumpv*(j_comp==1)
                                            + _lambda*(avgu_x*(k_comp==0) + avgu_y*(k_comp==1))*(n_x*jumpv*(j_comp==0) + n_y*jumpv*(j_comp==1)));
                                    }
                                    
                                }
                                else if (q_type==0)  { // edge on interior
                                    double adjacent_cell_local_index_q = adjacent_cell_local_index[q];
                                    //int current_edge_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                                    //int adjacent_cell_local_index = (int)(adjacent_elements(i, current_edge_num));

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
                                    if (i <= adjacent_cell_local_index_q) continue;
                                    
                                    TEUCHOS_ASSERT(adjacent_cell_local_index_q >= 0);
                                    int j_to_adjacent_cell = -1;
                                    if (particle_to_local_neighbor_lookup[adjacent_cell_local_index_q].count(particle_j)==1) {
                                        j_to_adjacent_cell = particle_to_local_neighbor_lookup[adjacent_cell_local_index_q][particle_j];
                                    }
                                    int k_to_adjacent_cell = -1;
                                    if (particle_to_local_neighbor_lookup[adjacent_cell_local_index_q].count(particle_k)==1) {
                                        k_to_adjacent_cell = particle_to_local_neighbor_lookup[adjacent_cell_local_index_q][particle_k];
                                    }
                                    if (j_to_adjacent_cell>=0) {
                                        j_has_value = true;
                                    }
                                    if (k_to_adjacent_cell>=0) {
                                        k_has_value = true;
                                    }
                                    if (j_to_adjacent_cell<0 && k_to_adjacent_cell<0 && j_to_cell_i<0 && k_to_cell_i<0) continue;
                                    
                                    auto normal_direction_correction = (i > adjacent_cell_local_index_q) ? 1 : -1; 
                                    auto n_x = normal_direction_correction * unit_normals(i,2*q+0);
                                    auto n_y = normal_direction_correction * unit_normals(i,2*q+1);
                                    // gives a unique normal that is always outward to the cell with the lower index

                                    double avgu_x=0, avgv_x=0, avgu_y=0, avgv_y=0;
                                    double jumpu = 0.0;
                                    double jumpv = 0.0;

                                    // gets quadrature # on adjacent cell (enumerates quadrature on 
                                    // side_of_cell_i_to_adjacent_cell in reverse due to orientation)
                                    int adjacent_q = num_interior_quadrature + side_of_cell_i_to_adjacent_cell[q]*num_exterior_quadrature_per_edge + (num_exterior_quadrature_per_edge - ((q-num_interior_quadrature)%num_exterior_quadrature_per_edge) - 1);


                                    //int adjacent_q = num_interior_quadrature + side_i_to_adjacent_cell*num_exterior_quadrature_per_edge + ((q-num_interior_quadrature)%num_exterior_quadrature_per_edge);

                                    //// diagnostic that quadrature matches up between adjacent_cell_local_index & adjacent_q 
                                    //// along with i & q
                                    //auto my_x = quadrature_points(i, 2*q+0);
                                    //auto my_wt = quadrature_weights(i, q);
                                    //auto their_wt = quadrature_weights(adjacent_cell_local_index, adjacent_q);
                                    //if (std::abs(my_wt-their_wt)>1e-14) printf("wm: %f, t: %f, d: %.16f\n", my_wt, their_wt, my_wt-their_wt);
                                    //auto my_y = quadrature_points(i, 2*q+1);
                                    //auto their_x = quadrature_points(adjacent_cell_local_index, 2*adjacent_q+0);
                                    //auto their_y = quadrature_points(adjacent_cell_local_index, 2*adjacent_q+1);
                                    //if (std::abs(my_x-their_x)>1e-14) printf("xm: %f, t: %f, d: %.16f\n", my_x, their_x, my_x-their_x);
                                    //if (std::abs(my_y-their_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_y, their_y, my_y-their_y);


                                    // diagnostics for quadrature points
                                    //printf("b: %f\n", _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index, j_to_adjacent_cell, adjacent_q+1));
                                    //auto my_j_index = _cell_particles_neighborhood->getNeighbor(i,j_to_cell_i);
                                    //auto their_j_index = _cell_particles_neighborhood->getNeighbor(adjacent_cell_local_index, j_to_adjacent_cell);
                                    //printf("xm: %d, t: %d\n", my_j_index, their_j_index);
                                    //auto my_q_x = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 0);
                                    //auto my_q_y = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 1);
                                    //auto their_q_x = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 0);
                                    //auto their_q_y = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 1);
                                    //if (std::abs(my_q_x-their_q_x)>1e-14) printf("xm: %f, t: %f, d: %.16f\n", my_q_x, their_q_x, my_q_x-their_q_x);
                                    //if (std::abs(my_q_y-their_q_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_q_y, their_q_y, my_q_y-their_q_y);

                                    //auto my_q_x = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 0);
                                    //auto my_q_y = _kokkos_quadrature_coordinates_host(i*_weights_ndim + q, 1);
                                    ////auto their_q_index = _kokkos_quadrature_neighbor_lists_host(adjacent_cell_local_index, 1+k_to_adjacent_cell);
                                    //auto their_q_x = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 0);
                                    //auto their_q_y = _kokkos_quadrature_coordinates_host(adjacent_cell_local_index*_weights_ndim + adjacent_q, 1);
                                    //if (std::abs(my_q_x-their_q_x)>1e-14) printf("xm: %f, t: %f, d: %.16f\n", my_q_x, their_q_x, my_q_x-their_q_x);
                                    //if (std::abs(my_q_y-their_q_y)>1e-14) printf("ym: %f, t: %f, d: %.16f\n", my_q_y, their_q_y, my_q_y-their_q_y);
 
                                    double other_v = (j_to_adjacent_cell>=0) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index_q, j_to_adjacent_cell, adjacent_q+1) : 0;
                                    jumpv = v-other_v;

                                    double other_u = (k_to_adjacent_cell>=0) ? _gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adjacent_cell_local_index_q, k_to_adjacent_cell, adjacent_q+1) : 0.0;
                                    jumpu = u-other_u;

                                    jumpu *= normal_direction_correction;
                                    jumpv *= normal_direction_correction;

                                    double other_grad_v_x = (j_to_adjacent_cell>=0) ? _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index_q, 0, j_to_adjacent_cell, adjacent_q+1) : 0.0;
                                    double other_grad_v_y = (j_to_adjacent_cell>=0) ? _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index_q, 1, j_to_adjacent_cell, adjacent_q+1) : 0.0;
                                    double other_grad_u_x = (k_to_adjacent_cell>=0) ? _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index_q, 0, k_to_adjacent_cell, adjacent_q+1) : 0.0;
                                    double other_grad_u_y = (k_to_adjacent_cell>=0) ? _gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, adjacent_cell_local_index_q, 1, k_to_adjacent_cell, adjacent_q+1) : 0.0;

                                    avgv_x = 0.5*(grad_v_x+other_grad_v_x);
                                    avgv_y = 0.5*(grad_v_y+other_grad_v_y);
                                    avgu_x = 0.5*(grad_u_x+other_grad_u_x);
                                    avgu_y = 0.5*(grad_u_y+other_grad_u_y);

                                    //double jump_u_jump_v = 0.0;
                                    //auto jump_v_x = n_x*v-n_x*other_v;
                                    //auto jump_v_y = n_y*v-n_y*other_v;
                                    //auto jump_u_x = n_x*u-n_x*other_u;
                                    //auto jump_u_y = n_y*u-n_y*other_u;
                                    //jump_u_jump_v = jump_v_x*jump_u_x + jump_v_y*jump_u_y;

                                    //contribution += 0.5 * penalty * q_wt * jumpv * jumpu; // other half will be added by other cell
                                    //contribution -= 0.5 * q_wt * _diffusion * (avgv_x * n_x + avgv_y * n_y) * jumpu;
                                    //contribution -= 0.5 * q_wt * _diffusion * (avgu_x * n_x + avgu_y * n_y) * jumpv;
                                    if (rd_op) {
                                        contribution += penalty * q_wt * jumpv * jumpu; // other half will be added by other cell
                                        contribution -= q_wt * _diffusion * (avgv_x * n_x + avgv_y * n_y) * jumpu;
                                        contribution -= q_wt * _diffusion * (avgu_x * n_x + avgu_y * n_y) * jumpv;
                                    } else if (le_op) {
                                        contribution += penalty * q_wt * jumpv*jumpu*(j_comp==k_comp); // other half will be added by other cell
                                        contribution -= q_wt * (
                                              2 * _shear * (n_x*avgv_x*(j_comp==0) 
                                                + 0.5*n_y*(avgv_y*(j_comp==0) + avgv_x*(j_comp==1))) * jumpu*(k_comp==0)  
                                            + 2 * _shear * (n_y*avgv_y*(j_comp==1) 
                                                + 0.5*n_x*(avgv_x*(j_comp==1) + avgv_y*(j_comp==0))) * jumpu*(k_comp==1)
                                            + _lambda*(avgv_x*(j_comp==0) + avgv_y*(j_comp==1))*(n_x*jumpu*(k_comp==0) + n_y*jumpu*(k_comp==1)));
                                        contribution -= q_wt * (
                                              2 * _shear * (n_x*avgu_x*(k_comp==0) 
                                                + 0.5*n_y*(avgu_y*(k_comp==0) + avgu_x*(k_comp==1))) * jumpv*(j_comp==0)  
                                            + 2 * _shear * (n_y*avgu_y*(k_comp==1) 
                                                + 0.5*n_x*(avgu_x*(k_comp==1) + avgu_y*(k_comp==0))) * jumpv*(j_comp==1)
                                            + _lambda*(avgu_x*(k_comp==0) + avgu_y*(k_comp==1))*(n_x*jumpv*(j_comp==0) + n_y*jumpv*(j_comp==1)));
                                    }
                                }
                                TEUCHOS_ASSERT(contribution==contribution);
                            }
                        }

                        val_data[0] = contribution;
                        if (j_has_value && k_has_value) {
                            this->_A->sumIntoLocalValues(row, 1, &val_data[0], &col_data[0], true);//, /*atomics*/false);
                        }
                    }
                }
            }
        }
    //}
    //area = t_area;
    //perimeter = t_perimeter;
    }, Kokkos::Sum<scalar_type>(area));


    // DIAGNOSTIC:: get global area
	scalar_type global_area;
	Teuchos::Ptr<scalar_type> global_area_ptr(&global_area);
	Teuchos::reduceAll<int, scalar_type>(*(_cells->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, area, global_area_ptr);
    if (_cells->getCoordsConst()->getComm()->getRank()==0) {
        printf("GLOBAL AREA: %.16f\n", global_area);
    }
	//scalar_type global_perimeter;
	//Teuchos::Ptr<scalar_type> global_perimeter_ptr(&global_perimeter);
	//Teuchos::reduceAll<int, scalar_type>(*(_cells->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, perimeter, global_perimeter_ptr);
    //if (_cells->getCoordsConst()->getComm()->getRank()==0) {
    //    printf("GLOBAL PERIMETER: %.16f\n", global_perimeter);///((double)(num_interior_edges)));
    //}

	TEUCHOS_ASSERT(!this->_A.is_null());
	ComputeMatrixTime->stop();

}

const std::vector<InteractingFields> ReactionDiffusionPhysics::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	return field_interactions;
}
    
}
