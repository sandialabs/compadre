#include <tpl/nanoflann.hpp>
#include <Kokkos_Core.hpp>
#include <assert.h>

namespace Compadre {

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
            assert((std::is_same<typename view_type::memory_space, Kokkos::HostSpace>::value) &&
                    "Views passed to PointCloudSearch at construction should reside on the host.");
            assert((src_pts_view.dimension_1()==3 && trg_pts_view.dimension_1()==3) &&
                    "Views passed to PointCloudSearch at construction must have second dimension of 3.");
        };

        PointCloudSearch(view_type src_pts_view) : PointCloudSearch(src_pts_view, src_pts_view) {};
    
        ~PointCloudSearch() {};


    
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
                const int dimension = 3, const double epsilon_multiplier = 1.2) {

            assert((std::is_same<typename neighbor_lists_view_type::memory_space, Kokkos::HostSpace>::value) &&
                    "Views passed to generateNeighborListsFromKNNSearch should reside on the host.");
            assert((std::is_same<typename epsilons_view_type::memory_space, Kokkos::HostSpace>::value) &&
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
            assert((neighbor_lists.dimension_0()==num_target_sites 
                        && neighbor_lists.dimension_1()==neighbors_needed+1) 
                        && "neighbor lists View does not have the correct dimensions");
            assert((epsilons.dimension_0()==num_target_sites)
                        && "epsilons View does not have the correct dimension");

            typedef Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
                    scratch_double_view;

            typedef Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
                    scratch_int_view;

            typedef Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace> loop_policy;
            typedef Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type loop_member_type;

            // determine scratch space size needed
            int team_scratch_size = 0;
            team_scratch_size += scratch_double_view::shmem_size(neighbors_needed); // distances
            team_scratch_size += scratch_int_view::shmem_size(neighbors_needed); // indices
            team_scratch_size += scratch_double_view::shmem_size(3); // target coordinate
            team_scratch_size += scratch_int_view::shmem_size(1); // neighbors found

            // each row of neighbor lists is a neighbor list for the target site corresponding to that row
            Kokkos::parallel_for(loop_policy(num_target_sites, Kokkos::AUTO)
                    .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
                    KOKKOS_LAMBDA(const loop_member_type& teamMember) {

                // make unmanaged scratch views
                scratch_double_view neighbor_distances(teamMember.team_scratch(0 /*shared memory*/), neighbors_needed);
                scratch_int_view neighbor_indices(teamMember.team_scratch(0 /*shared memory*/), neighbors_needed);
                scratch_double_view this_target_coord(teamMember.team_scratch(0 /*shared memory*/), 3);
                scratch_int_view neighbors_found(teamMember.team_scratch(0 /*shared memory*/), 1);

                const int i = teamMember.league_rank();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbors_needed), [=](const int j) {
                    neighbor_indices(j) = 0;
                    neighbor_distances(j) = 0;
                });
            
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    // target_coords is LayoutLeft on device and its HostMirror, so giving a pointer to 
                    // this data would lead to a wrong result if the device is a GPU

                    for (int j=0; j<3; ++j) this_target_coord(j) = _trg_pts_view(i,j);

                    neighbors_found(0) = kd_tree->knnSearch(this_target_coord.data(), neighbors_needed, 
                            neighbor_indices.data(), neighbor_distances.data()) ;
            
                    // the number of neighbors is stored in column zero of the neighbor lists 2D array
                    neighbor_lists(i,0) = neighbors_found(0);
            
                    // scale by epsilon_multiplier to window from location where the last neighbor was found
                    epsilons(i) = std::sqrt(neighbor_distances(neighbors_found(0)-1))*epsilon_multiplier;
                    // neighbor_distances stores squared distances from neighbor to target, as returned by nanoflann
                });

                // loop over each neighbor index and fill with a value
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbors_found(0)), [=](const int j) {
                    neighbor_lists(i,j+1) = neighbor_indices(j);
                });

            });

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
