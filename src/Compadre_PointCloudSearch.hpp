#include <tpl/nanoflann.hpp>
#include <Kokkos_Core.hpp>
#include "Compadre_Typedefs.hpp"
#include <memory>

namespace Compadre {

//! Custom RadiusResultSet for nanoflann that uses pre-allocated space for indices and radii
//! instead of using std::vec for std::pairs
template <typename _DistanceType, typename _IndexType = size_t>
class RadiusResultSet {

  public:

    typedef _DistanceType DistanceType;
    typedef _IndexType IndexType;

    const DistanceType radius;
    IndexType count;
    DistanceType* r_dist;
    IndexType* i_dist;
    const IndexType max_size;

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
            // would throw an exception here if count>=max_size, but this code is 
            // called inside of a parallel region so only an abort is possible, 
            // but this situation is recoverable 
            //
            // instead, we increase count, but there is nowhere to store neighbors
            // since we are working with pre-allocated space
            // this will be discovered after returning from the search by comparing
            // the count against the pre-allocate space dimensions
            if (count<max_size) {
                i_dist[count] = index;
                r_dist[count] = dist;
            }
            count++;
        }
        return true;

    }

    inline DistanceType worstDist() const { return radius; }

    std::pair<IndexType, DistanceType> worst_item() const {
        // just to verify this isn't ever called
        compadre_kernel_assert_release(false && "worst_item() should not be called.");
    }

    inline void sort() { 
        // puts closest neighbor as the first entry in the neighbor list
        // leaves the rest unsorted
 
        if (count > 0) {

            DistanceType best_distance = std::numeric_limits<DistanceType>::max();
            IndexType best_distance_index = 0;
            int best_index = -1;
            for (IndexType i=0; i<count; ++i) {
                if (r_dist[i] < best_distance) {
                    best_distance = r_dist[i];
                    best_distance_index = i_dist[i];
                    best_index = i;
                }
            }

            if (best_index != 0) {
                auto tmp_ind = i_dist[0];
                i_dist[0] = best_distance_index;
                i_dist[best_index] = tmp_ind;
            }
        }
    }
};


//! PointCloudSearch generates neighbor lists and window sizes for each target site
template <typename view_type>
class PointCloudSearch {

    protected:

        // define the cloud_type using this class
        typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloudSearch<view_type> >, 
                PointCloudSearch<view_type>, 3> tree_type;


        //! source site coordinates
        view_type _src_pts_view;

    public:

        PointCloudSearch(view_type src_pts_view) 
                : _src_pts_view(src_pts_view) {
            compadre_assert_release((std::is_same<typename view_type::memory_space, Kokkos::HostSpace>::value) &&
                    "Views passed to PointCloudSearch at construction should reside on the host.");
            compadre_assert_release(src_pts_view.extent(1)==3 &&
                    "Views passed to PointCloudSearch at construction must have second dimension of 3.");
        };
    
        ~PointCloudSearch() {};

        //! Returns a liberal estimated upper bound on number of neighbors to be returned by a neighbor search
        //! for a given choice of dimension, basis size, and epsilon_multiplier. Assumes quasiuniform distribution
        //! of points. This result can be used to size a preallocated neighbor_lists kokkos view.
        static inline int getEstimatedNumberNeighborsUpperBound(int unisolvency_size, const int dimension, const double epsilon_multiplier) {
            int multiplier = 1;
            if (dimension==1) multiplier = 2;
            return multiplier * 2.0 * unisolvency_size * pow(epsilon_multiplier, dimension) + 1; // +1 is for the number of neighbors entry needed in the first entry of each row
        }
    
        //! Bounding box query method required by Nanoflann.
        template <class BBOX> bool kdtree_get_bbox(BBOX& bb) const {return false;}

        //! Returns the number of source sites
        inline int kdtree_get_point_count() const {return _src_pts_view.extent(0);}

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

        std::shared_ptr<tree_type> generateKDTree(const int dimension = 3) {

            // maximum number of levels in the tree constructed by nanoflann
            const int maxLeaf = 10;

            // allocate a kd_tree from nanoflann
            std::shared_ptr<tree_type> kd_tree(new tree_type(dimension, *this, nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf)));

            // insert points to build up the tree
            kd_tree->buildIndex();

            return kd_tree;

        }

        //! Generates neighbor lists
        template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
        void generateNeighborListsFromKNNSearch(trg_view_type trg_pts_view, neighbor_lists_view_type neighbor_lists,
                epsilons_view_type epsilons, const int neighbors_needed, 
                const int dimension = 3, const double epsilon_multiplier = 1.6, std::shared_ptr<tree_type> kd_tree = NULL, bool max_search_radius = 0.0) {

            // First, do a knn search (removes need for guessing initial search radius)

            compadre_assert_release((std::is_same<typename trg_view_type::memory_space, Kokkos::HostSpace>::value) &&
                    "Target coordinates view passed to generateNeighborListsFromKNNSearch should reside on the host.");
            compadre_assert_release(trg_pts_view.extent(1)==3 &&
                    "Target coordinates view passed to generateNeighborListsFromKNNSearch must have second dimension of 3.");
            compadre_assert_release((std::is_same<typename neighbor_lists_view_type::memory_space, Kokkos::HostSpace>::value) &&
                    "Views passed to generateNeighborListsFromKNNSearch should reside on the host.");
            compadre_assert_release((std::is_same<typename epsilons_view_type::memory_space, Kokkos::HostSpace>::value) &&
                    "Views passed to generateNeighborListsFromKNNSearch should reside on the host.");

            // loop size
            const int num_target_sites = trg_pts_view.extent(0);

            if (!kd_tree)
                kd_tree = this->generateKDTree(dimension);

            // allocate neighbor lists and epsilons
            compadre_assert_release((neighbor_lists.extent(0)==(size_t)num_target_sites 
                        && neighbor_lists.extent(1)>=(size_t)(neighbors_needed+1)) 
                        && "neighbor lists View does not have large enough dimensions");
            compadre_assert_release((epsilons.extent(0)==(size_t)num_target_sites)
                        && "epsilons View does not have the correct dimension");

            typedef Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
                    scratch_double_view;

            typedef Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
                    scratch_int_view;

            // determine scratch space size needed
            int team_scratch_size = 0;
            team_scratch_size += scratch_double_view::shmem_size(neighbor_lists.extent(1)); // distances
            team_scratch_size += scratch_int_view::shmem_size(neighbor_lists.extent(1)); // indices
            team_scratch_size += scratch_double_view::shmem_size(3); // target coordinate

            // minimum number of neighbors found over all target sites' neighborhoods
            size_t min_num_neighbors = 0;
            //
            // part 1. do knn search for neighbors needed for unisolvency
            // each row of neighbor lists is a neighbor list for the target site corresponding to that row
            //
            compadre_assert_release((neighbor_lists.extent(1) >= (neighbors_needed+1)) && "neighbor_lists does not contain enough columns for the minimum number of neighbors needed to be stored.");
            // as long as neighbor_lists can hold the number of neighbors_needed, we don't need to check
            // that the maximum number of neighbors will fit into neighbor_lists
            //
            Kokkos::parallel_reduce("knn search", host_team_policy(num_target_sites, Kokkos::AUTO)
                    .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
                    KOKKOS_LAMBDA(const host_member_type& teamMember, size_t& t_min_num_neighbors) {

                // make unmanaged scratch views
                scratch_double_view neighbor_distances(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.extent(1));
                scratch_int_view neighbor_indices(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.extent(1));
                scratch_double_view this_target_coord(teamMember.team_scratch(0 /*shared memory*/), 3);

                size_t neighbors_found = 0;

                const int i = teamMember.league_rank();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbor_lists.extent(1)), [=](const int j) {
                    neighbor_indices(j) = 0;
                    neighbor_distances(j) = 0;
                });
            
                teamMember.team_barrier();
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    // target_coords is LayoutLeft on device and its HostMirror, so giving a pointer to 
                    // this data would lead to a wrong result if the device is a GPU

                    for (int j=0; j<3; ++j) this_target_coord(j) = trg_pts_view(i,j);

                    neighbors_found = kd_tree->knnSearch(this_target_coord.data(), neighbors_needed, 
                            neighbor_indices.data(), neighbor_distances.data()) ;
            
                    // get minimum number of neighbors found over all target sites' neighborhoods
                    t_min_num_neighbors = std::min(neighbors_found, t_min_num_neighbors);
            
                    // scale by epsilon_multiplier to window from location where the last neighbor was found
                    epsilons(i) = std::sqrt(neighbor_distances(neighbors_found-1))*epsilon_multiplier;

                    // needs furthest neighbor's distance for next portion
                    compadre_kernel_assert_release((epsilons(i)<=max_search_radius || max_search_radius==0) && "max_search_radius given (generally derived from the size of a halo region), and search radius needed would exceed this max_search_radius.");
                    // neighbor_distances stores squared distances from neighbor to target, as returned by nanoflann
                });
            }, Kokkos::Min<size_t>(min_num_neighbors) );
            Kokkos::fence();

            // Next, check that we found the neighbors_needed number that we require for unisolvency
            compadre_assert_release((min_num_neighbors>=neighbors_needed) 
                    && "Neighbor search failed to find number of neighbors needed for unisolvency.");
            

            // determine scratch space size needed
            team_scratch_size = 0;
            team_scratch_size += scratch_double_view::shmem_size(neighbor_lists.extent(1)); // distances
            team_scratch_size += scratch_int_view::shmem_size(neighbor_lists.extent(1)); // indices
            team_scratch_size += scratch_double_view::shmem_size(3); // target coordinate

            // maximum number of neighbors found over all target sites' neighborhoods
            size_t max_num_neighbors = 0;
            // part 2. do radius search using window size from knn search
            // each row of neighbor lists is a neighbor list for the target site corresponding to that row
            Kokkos::parallel_reduce("radius search", host_team_policy(num_target_sites, Kokkos::AUTO)
                    .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
                    KOKKOS_LAMBDA(const host_member_type& teamMember, size_t& t_max_num_neighbors) {

                // make unmanaged scratch views
                scratch_double_view neighbor_distances(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.extent(1));
                scratch_int_view neighbor_indices(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.extent(1));
                scratch_double_view this_target_coord(teamMember.team_scratch(0 /*shared memory*/), 3);

                size_t neighbors_found;

                const int i = teamMember.league_rank();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbor_lists.extent(1)), [=](const int j) {
                    neighbor_indices(j) = 0;
                    neighbor_distances(j) = 0;
                });
            
                teamMember.team_barrier();

                auto data_for_search = std::vector<std::pair<unsigned long int, double> >();

                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    // target_coords is LayoutLeft on device and its HostMirror, so giving a pointer to 
                    // this data would lead to a wrong result if the device is a GPU

                    for (int j=0; j<3; ++j) this_target_coord(j) = trg_pts_view(i,j);

                    nanoflann::SearchParams sp; // default parameters
                    Compadre::RadiusResultSet<double> rrs(epsilons(i)*epsilons(i), neighbor_distances.data(), neighbor_indices.data(), neighbor_lists.extent(1));
                    neighbors_found = kd_tree->template radiusSearchCustomCallback<Compadre::RadiusResultSet<double> >(this_target_coord.data(), rrs, sp) ;
                    t_max_num_neighbors = std::max(neighbors_found, t_max_num_neighbors);
            
                    // the number of neighbors is stored in column zero of the neighbor lists 2D array
                    neighbor_lists(i,0) = neighbors_found;

                    // epsilons already scaled and then set by search radius
                });
                teamMember.team_barrier();

                // loop_bound so that we don't write into memory we don't have allocated
                int loop_bound = std::min(neighbors_found, neighbor_lists.extent(1)-1);
                // loop over each neighbor index and fill with a value
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, loop_bound), [&](const int j) {
                    // cast to an whatever data type the 2D array of neighbor lists is using
                    neighbor_lists(i,j+1) = static_cast<typename std::remove_pointer<typename std::remove_pointer<typename neighbor_lists_view_type::data_type>::type>::type>(neighbor_indices(j));
                });

            }, Kokkos::Max<size_t>(max_num_neighbors) );
            Kokkos::fence();

            // check if max_num_neighbors will fit onto pre-allocated space
            compadre_assert_release((neighbor_lists.extent(1) >= (max_num_neighbors+1)) 
                    && "neighbor_lists does not contain enough columns for the maximum number of neighbors needing to be stored.");


        }

}; // PointCloudSearch

//! CreatePointCloudSearch allows for the construction of an object of type PointCloudSearch with template deduction
template <typename view_type>
PointCloudSearch<view_type> CreatePointCloudSearch(view_type src_view) { 
    return PointCloudSearch<view_type>(src_view);
}

}; // Compadre
