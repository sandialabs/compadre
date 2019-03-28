#include <tpl/nanoflann.hpp>
#include <Kokkos_Core.hpp>
#include "Compadre_Typedefs.hpp"

namespace Compadre {

//! Custom RadiusResultSet for nanoflann that uses pre-allocated space for indices and radii
//! instead of using std::vec for std::pairs
template <typename _DistanceType, typename _IndexType = size_t>
class RadiusResultSet {

  public:

    typedef _DistanceType DistanceType;
    typedef _IndexType IndexType;

    const DistanceType radius;
    const IndexType max_size;
    IndexType count;
    DistanceType* r_dist;
    IndexType* i_dist;

    inline RadiusResultSet(
        DistanceType radius_,
        DistanceType* r_dist_, IndexType* i_dist_, const IndexType max_size_)
        : radius(radius_), r_dist(r_dist_), i_dist(i_dist_), max_size(max_size_) {
        count = 0;
        init();
    }

    inline void init() {}

    inline void clear() { count = 0; }

    inline size_t size() const { return count; }

    inline bool full() const { return true; }

    inline bool addPoint(DistanceType dist, IndexType index) {

        if (dist < radius) {
            compadre_kernel_assert_release((count<max_size) && "Neighbors found exceeds space allocated for their storage.");
            i_dist[count] = index;
            r_dist[count] = dist;
            count++;
        }
        return true;

    }

    inline DistanceType worstDist() const { return radius; }

    std::pair<IndexType, DistanceType> worst_item() const {

        compadre_kernel_assert_release((count>0) && "Cannot invoke RadiusResultSet::worst_item() on "
                                                    "an empty list of results.");

        DistanceType worst_distance = std::numeric_limits<DistanceType>::max();
        IndexType worst_distance_index = 0;
        for (int i=0; i<=count; ++i) {
            if (r_dist[i] < worst_distance) {
                worst_distance = r_dist[i];
                worst_distance_index = i_dist[i];
            }
        }

        auto worst_pair = std::pair<IndexType, DistanceType>(worst_distance_index, worst_distance);
        return worst_pair;

    }

    inline void sort() { 
        // puts closest neighbor as the first entry in the neighbor list

        if (count > 0) {

            DistanceType worst_distance = std::numeric_limits<DistanceType>::max();
            IndexType worst_distance_index = 0;
            IndexType worst_index = 0;
            for (int i=0; i<=count; ++i) {
                if (r_dist[i] < worst_distance) {
                    worst_distance = r_dist[i];
                    worst_distance_index = i_dist[i];
                    worst_index = i;
                }
            }

            if (worst_index != 0) {
                auto tmp_ind = i_dist[0];
                i_dist[0] = worst_distance_index;
                i_dist[worst_index] = tmp_ind;
            }

        }
    }
};


//! PointCloudSearch generates neighbor lists and window sizes for each target site
template <typename view_type>
class PointCloudSearch {

    protected:

        //! source site coordinates
        view_type _src_pts_view;

        //! target site coordinates
        view_type _trg_pts_view;

    public:

        PointCloudSearch(view_type src_pts_view, view_type trg_pts_view) 
                : _src_pts_view(src_pts_view), _trg_pts_view(trg_pts_view) {
            compadre_assert_release((std::is_same<typename view_type::memory_space, Kokkos::HostSpace>::value) &&
                    "Views passed to PointCloudSearch at construction should reside on the host.");
            compadre_assert_release((src_pts_view.dimension_1()==3 && trg_pts_view.dimension_1()==3) &&
                    "Views passed to PointCloudSearch at construction must have second dimension of 3.");
        };

        PointCloudSearch(view_type src_pts_view) : PointCloudSearch(src_pts_view, src_pts_view) {};
    
        ~PointCloudSearch() {};

        //! Returns a liberal estimated upper bound on number of neighbors to be returned by a neighbor search
        //! for a given choice of dimension, basis size, and epsilon_multiplier. Assumes quasiuniform distribution
        //! of points. This result can be used to size a preallocated neighbor_lists kokkos view.
        static inline int getEstimatedNumberNeighborsUpperBound(int unisolvency_size, const int dimension, const double epsilon_multiplier) {
            return 2.0 * unisolvency_size * pow(epsilon_multiplier, dimension) + 1; // +1 is for the number of neighbors entry needed in the first entry of each row
        }
    
        //! Bounding box query method required by Nanoflann.
        template <class BBOX> bool kdtree_get_bbox(BBOX& bb) const {return false;}

        //! Returns the number of source sites
        inline int kdtree_get_point_count() const {return _src_pts_view.dimension_0();}

        //! Returns the coordinate value of a point
        inline double kdtree_get_pt(const int idx, int dim) const {return _src_pts_view(idx,dim);}

        //! Returns the distance between a point and a source site, given its index
        inline double kdtree_distance(const double* queryPt, const int idx, long long sz) const {

            double distance = 0;
            for (int i=0; i<3; ++i) {
                distance += (_src_pts_view(idx,i)-queryPt[i])*(_src_pts_view(idx,i)-queryPt[i]);
            }
            return std::sqrt(distance);

        }

        //! Generates neighbor lists
        template <typename neighbor_lists_view_type, typename epsilons_view_type>
        void generateNeighborListsFromKNNSearch(neighbor_lists_view_type neighbor_lists,
                epsilons_view_type epsilons, const int neighbors_needed, 
                const int dimension = 3, const double epsilon_multiplier = 1.6, bool max_search_radius = 0.0) {

            compadre_assert_release((std::is_same<typename neighbor_lists_view_type::memory_space, Kokkos::HostSpace>::value) &&
                    "Views passed to generateNeighborListsFromKNNSearch should reside on the host.");
            compadre_assert_release((std::is_same<typename epsilons_view_type::memory_space, Kokkos::HostSpace>::value) &&
                    "Views passed to generateNeighborListsFromKNNSearch should reside on the host.");

            // define the cloud_type using this class
            typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, decltype(*this)>, 
                    decltype(*this), 3> tree_type;

            // default search parameters
            nanoflann::SearchParams search_parameters; 

            // maximum number of levels in the tree constructed by nanoflann
            const int maxLeaf = 10;

            // allocate a kd_tree from nanoflann
            tree_type *kd_tree = new tree_type(dimension, *this, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf));

            // insert points to build up the tree
            kd_tree->buildIndex();

            // loop size
            const int num_target_sites = _trg_pts_view.dimension_0();

            // allocate neighbor lists and epsilons
            compadre_assert_release((neighbor_lists.dimension_0()==num_target_sites 
                        && neighbor_lists.dimension_1()>=neighbors_needed+1) 
                        && "neighbor lists View does not have large enough dimensions");
            compadre_assert_release((epsilons.dimension_0()==num_target_sites)
                        && "epsilons View does not have the correct dimension");

            typedef Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
                    scratch_double_view;

            typedef Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
                    scratch_int_view;

            typedef Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace> loop_policy;
            typedef Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type loop_member_type;

            // determine scratch space size needed
            int team_scratch_size = 0;
            team_scratch_size += scratch_double_view::shmem_size(neighbor_lists.dimension_1()); // distances
            team_scratch_size += scratch_int_view::shmem_size(neighbor_lists.dimension_1()); // indices
            team_scratch_size += scratch_double_view::shmem_size(3); // target coordinate
            team_scratch_size += scratch_int_view::shmem_size(1); // neighbors found

            // part 1. do knn search for neighbors needed for unisolvency
            // each row of neighbor lists is a neighbor list for the target site corresponding to that row
            Kokkos::parallel_for(loop_policy(num_target_sites, Kokkos::AUTO)
                    .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
                    KOKKOS_LAMBDA(const loop_member_type& teamMember) {

                // make unmanaged scratch views
                scratch_double_view neighbor_distances(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.dimension_1());
                scratch_int_view neighbor_indices(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.dimension_1());
                scratch_double_view this_target_coord(teamMember.team_scratch(0 /*shared memory*/), 3);
                scratch_int_view neighbors_found(teamMember.team_scratch(0 /*shared memory*/), 1);

                const int i = teamMember.league_rank();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbor_lists.dimension_1()), [=](const int j) {
                    neighbor_indices(j) = 0;
                    neighbor_distances(j) = 0;
                });
            
                teamMember.team_barrier();
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    // target_coords is LayoutLeft on device and its HostMirror, so giving a pointer to 
                    // this data would lead to a wrong result if the device is a GPU

                    for (int j=0; j<3; ++j) this_target_coord(j) = _trg_pts_view(i,j);

                    neighbors_found(0) = kd_tree->knnSearch(this_target_coord.data(), neighbors_needed, 
                            neighbor_indices.data(), neighbor_distances.data()) ;
            
                    // the number of neighbors is stored in column zero of the neighbor lists 2D array
                    neighbor_lists(i,0) = neighbors_found(0);
                    compadre_kernel_assert_release((neighbor_lists(i,0)>=neighbors_needed) && "Neighbor search failed to find number of neighbors needed for unisolvency.");
            
                    // scale by epsilon_multiplier to window from location where the last neighbor was found
                    epsilons(i) = std::sqrt(neighbor_distances(neighbors_found(0)-1))*epsilon_multiplier;
                    compadre_kernel_assert_release((epsilons(i)<=max_search_radius || max_search_radius==0) && "max_search_radius given (generally derived from the size of a halo region), and search radius needed would exceed this max_search_radius.");
                    // neighbor_distances stores squared distances from neighbor to target, as returned by nanoflann
                });
            });
            Kokkos::fence();

            

            // determine scratch space size needed
            team_scratch_size = 0;
            team_scratch_size += scratch_double_view::shmem_size(neighbor_lists.dimension_1()); // distances
            team_scratch_size += scratch_int_view::shmem_size(neighbor_lists.dimension_1()); // indices
            team_scratch_size += scratch_double_view::shmem_size(3); // target coordinate
            team_scratch_size += scratch_int_view::shmem_size(1); // neighbors found

            // part 2. do radius search using window size from knn search
            // each row of neighbor lists is a neighbor list for the target site corresponding to that row
            Kokkos::parallel_for(loop_policy(num_target_sites, Kokkos::AUTO)
                    .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
                    KOKKOS_LAMBDA(const loop_member_type& teamMember) {

                // make unmanaged scratch views
                scratch_double_view neighbor_distances(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.dimension_1());
                scratch_int_view neighbor_indices(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.dimension_1());
                scratch_double_view this_target_coord(teamMember.team_scratch(0 /*shared memory*/), 3);
                scratch_int_view neighbors_found(teamMember.team_scratch(0 /*shared memory*/), 1);

                const int i = teamMember.league_rank();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbor_lists.dimension_1()), [=](const int j) {
                    neighbor_indices(j) = 0;
                    neighbor_distances(j) = 0;
                });
            
                teamMember.team_barrier();

                auto data_for_search = std::vector<std::pair<unsigned long int, double> >();

                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    // target_coords is LayoutLeft on device and its HostMirror, so giving a pointer to 
                    // this data would lead to a wrong result if the device is a GPU

                    for (int j=0; j<3; ++j) this_target_coord(j) = _trg_pts_view(i,j);

                    nanoflann::SearchParams sp; // default parameters
                    Compadre::RadiusResultSet<double> rrs(epsilons(i)*epsilons(i), neighbor_distances.data(), neighbor_indices.data(), neighbor_lists.dimension_1());
                    neighbors_found(0) = kd_tree->template radiusSearchCustomCallback<Compadre::RadiusResultSet<double> >(this_target_coord.data(), rrs, sp) ;
            
                    // the number of neighbors is stored in column zero of the neighbor lists 2D array
                    neighbor_lists(i,0) = neighbors_found(0);

                    // epsilons already set by search radius
                });

                // this is a check to make sure we have enough room to store all of the neighbors found
                // strictly less than in assertion because we need 1 entry for the total number of neighbors found in addition to their indices
                compadre_kernel_assert_release((neighbors_found(0)<neighbor_lists.dimension_1()) && "neighbor_lists given to PointCloudSearch has too few columns (second dimension) to hold all neighbors found.");

                teamMember.team_barrier();

                // loop over each neighbor index and fill with a value
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, static_cast<int>(neighbors_found(0))), [&](const int j) {
                    // cast to an whatever data type the 2D array of neighbor lists is using
                    neighbor_lists(i,j+1) = static_cast<typename std::remove_pointer<typename std::remove_pointer<typename neighbor_lists_view_type::data_type>::type>::type>(neighbor_indices(j));
                });

            });
            Kokkos::fence();

            // deallocate nanoflann kdtree
            free(kd_tree);
        }

}; // PointCloudSearch

//! CreatePointCloudSearch allows for the construction of an object of type PointCloudSearch with template deduction
template <typename view_type>
PointCloudSearch<view_type> CreatePointCloudSearch(view_type src_view, view_type trg_view) { 
    return PointCloudSearch<view_type>(src_view, trg_view);
}

//! CreatePointCloudSearch allows for the construction of an object of type PointCloudSearch with template deduction
template <typename view_type>
PointCloudSearch<view_type> CreatePointCloudSearch(view_type src_view) { 
    return PointCloudSearch<view_type>(src_view);
}

}; // Compadre
