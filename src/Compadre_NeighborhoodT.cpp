#include "Compadre_NeighborhoodT.hpp"

#include "Compadre_XyzVector.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"

#ifdef COMPADREHARNESS_USE_OPENMP
#include <omp.h>
#endif

namespace Compadre {

typedef Compadre::XyzVector xyz_type;
typedef Compadre::CoordsT coords_type;
typedef Compadre::CoordsT particles_type;

// protected 
NeighborhoodT::NeighborhoodT(const local_index_type dim) : _n_dim(dim), _local_max_num_neighbors(0), _storage_multiplier(1) {};

NeighborhoodT::NeighborhoodT(const particles_type* source_particles, const particles_type* target_particles, const bool use_physical_coords, const local_index_type max_leaf) : 
        _source_coords(source_particles->getCoordsConst()), _source_particles(source_particles), 
        _n_dim(_source_coords->nDim()), _local_max_num_neighbors(0), _storage_multiplier(1)
{
    if (target_particles) {
        _target_particles = target_particles;
        _target_coords = target_particles->getCoordsConst();
    } else {
        _target_particles = source_particles;
        _target_coords = source_particles->getCoordsConst();
    }

    // initialize h vector of window sizes
    const bool setToZero = true;
    _h_support_size = Teuchos::rcp(new mvec_type(_target_coords->getMapConst(), 1, setToZero));

    _kokkos_augmented_source_coordinates_host = host_view_scalar_type("source_coordinates", _source_coords->nLocal(true /* include halo in count */), _n_dim);
    Kokkos::deep_copy(_kokkos_augmented_source_coordinates_host, 0.0);

    // fill in the source coords, adding regular with halo coordiantes into a kokkos view
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int i) {
        xyz_type coordinate = _source_coords->getLocalCoords(i, true /*include halo*/, use_physical_coords);
        for (local_index_type j=0; j<_n_dim; ++j) {
            _kokkos_augmented_source_coordinates_host(i,j) = coordinate[j];
        }
    });
    _point_cloud_search = std::shared_ptr<Compadre::PointCloudSearch<host_view_scalar_type> >
        (new Compadre::PointCloudSearch<host_view_scalar_type>(_kokkos_augmented_source_coordinates_host, _n_dim, max_leaf));
    _point_cloud_search->generateKDTree();


    NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Neighbor Search Time");
    MaxNeighborComputeTime = Teuchos::TimeMonitor::getNewCounter ("Max Neighbor Compute Time");
}

void NeighborhoodT::setHSupportSize(const local_index_type idx, const scalar_type val) {
    host_view_type hsupportView = _h_support_size->getLocalView<host_view_type>();
    hsupportView(idx,0) = val;
}

double NeighborhoodT::getHSupportSize(const local_index_type idx) const {
    host_view_type hsupportView = _h_support_size->getLocalView<host_view_type>();
    return hsupportView(idx,0);
}

void NeighborhoodT::setAllHSupportSizes(const scalar_type val) {
    host_view_type hsupportView = _h_support_size->getLocalView<host_view_type>();
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,hsupportView.extent(0)), KOKKOS_LAMBDA(const int j) {
        hsupportView(j,0) = val;
    });
}

local_index_type NeighborhoodT::computeMaxNumNeighbors(const bool global) const {
    if (!global) {
        return _local_max_num_neighbors;
    } else {
        local_index_type global_processor_max_num_neighbors;
        Teuchos::Ptr<local_index_type> global_processor_max_num_neighbors_ptr(&global_processor_max_num_neighbors);
        Teuchos::reduceAll<local_index_type, local_index_type>(*(_target_coords->getComm()), Teuchos::REDUCE_MAX,
                _local_max_num_neighbors, global_processor_max_num_neighbors_ptr);
        return global_processor_max_num_neighbors;
    }
}

scalar_type NeighborhoodT::computeMinHSupportSize(const bool global) const {

    host_view_type hsupportView = _h_support_size->getLocalView<host_view_type>();
    scalar_type local_processor_min_radius;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_coords->nLocal()),
            KOKKOS_LAMBDA (const int i, scalar_type &myVal) {
        myVal = hsupportView(i,0)<myVal ? hsupportView(i,0) : myVal;
    }, Kokkos::Min<scalar_type>(local_processor_min_radius));

    if (global) {
        // collect min radius from all processors
        scalar_type global_processor_min_radius;
        Teuchos::Ptr<scalar_type> global_processor_min_radius_ptr(&global_processor_min_radius);
        Teuchos::reduceAll<local_index_type, scalar_type>(*(_target_coords->getComm()), Teuchos::REDUCE_MIN,
                local_processor_min_radius, global_processor_min_radius_ptr);

        return global_processor_min_radius;
    } else {
        return local_processor_min_radius;
    }

}

scalar_type NeighborhoodT::computeMaxHSupportSize(const bool global) const {

    host_view_type hsupportView = _h_support_size->getLocalView<host_view_type>();
    scalar_type local_processor_max_radius;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_coords->nLocal()),
            KOKKOS_LAMBDA (const int i, scalar_type &myVal) {
        myVal = hsupportView(i,0)>myVal ? hsupportView(i,0) : myVal;
    }, Kokkos::Max<scalar_type>(local_processor_max_radius));

    if (global) {
        // collect max radius from all processors
        scalar_type global_processor_max_radius;
        Teuchos::Ptr<scalar_type> global_processor_max_radius_ptr(&global_processor_max_radius);
        Teuchos::reduceAll<local_index_type, scalar_type>(*(_target_coords->getComm()), Teuchos::REDUCE_MAX,
                local_processor_max_radius, global_processor_max_radius_ptr);

        return global_processor_max_radius;
    } else {
        return local_processor_max_radius;
    }

}


void NeighborhoodT::constructAllNeighborLists(const scalar_type halo_max_search_size, std::string search_type, 
        bool do_dry_run_for_sizes, const local_index_type knn_needed, const scalar_type cutoff_multiplier, 
        const scalar_type radius, const bool equal_radii, const scalar_type post_search_radii_scaling, bool use_physical_coords) {

    NeighborSearchTime->start();

    const local_index_type comm_size = _source_coords->getComm()->getSize();
    bool global_reduction = (comm_size>1); // used for max num neighbors

    scalar_type recalculated_halo_max_search_size = (global_reduction) ? halo_max_search_size : 0.0;

    // get view from h_support and check its size
    host_view_type h_support_view = _h_support_size->getLocalView<host_view_type>();
    // should be zero or exactly as large as target coords
    // if zero, then allocate to size of target coords
    if (h_support_view.extent(0)==0) {
        // initialize h vector of window sizes
        const bool setToZero = true;
        _h_support_size = Teuchos::rcp(new mvec_type(_target_coords->getMapConst(), 1, setToZero));
        h_support_view = _h_support_size->getLocalView<host_view_type>();
    }
    // if search_type is "radius" and radius==0.0, then whatever is in h_support_size will be used
    if (search_type=="radius" && radius!=0.0) {
        this->setAllHSupportSizes(radius);
    }
    auto epsilons = host_vector_scalar_type(h_support_view.data(), h_support_view.extent(0));

    // get kokkos view of target points
    host_view_type target_pts_view = _target_coords->getPts(false/*halo*/, use_physical_coords)->getLocalView<host_view_type>();

    // get search_type to lowercase
    transform(search_type.begin(), search_type.end(), search_type.begin(), ::tolower);

    // do dry run to figure out how many neighbors are needed
    // optimal for storage, not for computation (2x the work)
    if (do_dry_run_for_sizes) {
        _neighbor_lists = host_view_local_index_type("neighbor lists", h_support_view.extent(0), 1);

        size_t max_num_neighbors;
        if (search_type=="knn") {
            _neighbor_lists = host_view_local_index_type("neighbor lists", h_support_view.extent(0), 1+knn_needed);
            max_num_neighbors = _point_cloud_search->generateNeighborListsFromKNNSearch(do_dry_run_for_sizes, 
                    target_pts_view, _neighbor_lists, epsilons, knn_needed, cutoff_multiplier, 
                    recalculated_halo_max_search_size);
        } else if (search_type=="radius") {
            _neighbor_lists = host_view_local_index_type("neighbor lists", h_support_view.extent(0), 1);
            max_num_neighbors = _point_cloud_search->generateNeighborListsFromRadiusSearch(do_dry_run_for_sizes, 
                    target_pts_view, _neighbor_lists, epsilons, radius, recalculated_halo_max_search_size);
        }

        // at this point, h_supports contains all search sizes
        // if needed, do a follow up radius search to ensure uniform search radii
        if (equal_radii) {
            // get max h size
            auto max_h = this->computeMaxHSupportSize(global_reduction);

            // perform a radius search with max_h
            max_num_neighbors = _point_cloud_search->generateNeighborListsFromRadiusSearch(do_dry_run_for_sizes, 
                    target_pts_view, _neighbor_lists, epsilons, max_h, recalculated_halo_max_search_size);
        }

        // follow up by figuring out max num neighbors resize neighbor_lists to that large
        Kokkos::resize(_neighbor_lists, _neighbor_lists.extent(0), 1+max_num_neighbors);
    } else {
        // use storage multiplier times estimated number to figure out sizes
        int estimated_upper_bound_number_neighbors = 
            (int)(_storage_multiplier*_point_cloud_search->getEstimatedNumberNeighborsUpperBound(knn_needed, 
                        _n_dim, cutoff_multiplier));
        _neighbor_lists = host_view_local_index_type("neighbor lists", h_support_view.extent(0), 
                estimated_upper_bound_number_neighbors);
    }

    // neighbor search with storage
    // determine which type of search to perform and take into consideration constraints
    if (search_type=="knn") {
        _local_max_num_neighbors = _point_cloud_search->generateNeighborListsFromKNNSearch(false /*not dry run*/, 
                target_pts_view, _neighbor_lists, epsilons, knn_needed, cutoff_multiplier, 
                recalculated_halo_max_search_size);
    } else if (search_type=="radius") {
        _local_max_num_neighbors = _point_cloud_search->generateNeighborListsFromRadiusSearch(false /*not dry run*/, 
                target_pts_view, _neighbor_lists, epsilons, radius, recalculated_halo_max_search_size);
    }

    // at this point, h_supports contains all search sizes
    // if needed, do a follow up radius search to ensure uniform search radii
    if (equal_radii) {
        // get max h size
        auto max_h = this->computeMaxHSupportSize(global_reduction);

        // perform a radius search with max_h
        _local_max_num_neighbors = _point_cloud_search->generateNeighborListsFromRadiusSearch(false /*not dry run*/, 
                target_pts_view, _neighbor_lists, epsilons, max_h, recalculated_halo_max_search_size);
    }

    if (post_search_radii_scaling != 1.0) {
        h_support_view = _h_support_size->getLocalView<host_view_type>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,h_support_view.extent(0)), KOKKOS_LAMBDA(const int i) {
            h_support_view(i,0) *= post_search_radii_scaling;
        });
    }

    NeighborSearchTime->stop();
}

}
