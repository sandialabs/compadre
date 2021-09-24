#ifndef _COMPADRE_POINTCLOUDSEARCH_HPP_
#define _COMPADRE_POINTCLOUDSEARCH_HPP_

#include "Compadre_Typedefs.hpp"
#include "Compadre_NeighborLists.hpp"
#include <Kokkos_Core.hpp>
#include <ArborX_LinearBVH.hpp>

template <typename memory_space>
using BVH = ArborX::BoundingVolumeHierarchy<memory_space>;

namespace Compadre {

//!  PointCloudSearch generates neighbor lists and window sizes for each target site
/*!
*  Search methods can be run in dry-run mode, or not.
*
*  #### When in dry-run mode:
*                                                                                                                         
*    `neighbors_list` will be populated with number of neighbors found for each target site.
*
*    This allows a user to know memory allocation needed before storage of neighbor indices.
*                                                                                                                         
*  #### When not in dry-run mode:
*
*    `neighbors_list_offsets` will be populated with offsets for values (dependent on method) determined by neighbor_list.
*    If a 2D view for `neighbors_list` is used, then \f$ N(i,j+1) \f$ will store the \f$ j^{th} \f$ neighbor of \f$ i \f$,
*    and \f$ N(i,0) \f$ will store the number of neighbors for target \f$ i \f$.
*
*/
template <typename view_type>
class PointCloudSearch {

    protected:

        //! source site coordinates
        view_type _src_pts_view;
        local_index_type _dim;
        local_index_type _max_leaf;

        BVH<device_memory_space> _tree;

    public:

        PointCloudSearch(view_type src_pts_view, const local_index_type dimension = -1,
                const local_index_type max_leaf = -1) 
                : _src_pts_view(src_pts_view), 
                  _dim((dimension < 0) ? src_pts_view.extent(1) : dimension),
                  _max_leaf((max_leaf < 0) ? 10 : max_leaf) {
            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename view_type::memory_space>::accessible==1)
                    && "Views passed to PointCloudSearch at construction should be accessible from the host.");

            { //TODO: Add check here to see if _tree is populated
                // if not, populate
                Kokkos::View<ArborX::Point *> cloud("point_cloud", _src_pts_view.extent(0));
                Kokkos::parallel_for(_src_pts_view.extent(0), KOKKOS_LAMBDA(int i) {
                  if (_dim==1) {
                      cloud(i) = {{_src_pts_view(i,0),0,0}};
                  }
                  if (_dim==2) {
                      cloud(i) = {{_src_pts_view(i,0),_src_pts_view(i,1),0}};
                  }
                  if (_dim==3) {
                      cloud(i) = {{_src_pts_view(i,0),_src_pts_view(i,1),_src_pts_view(i,2)}};
                  }
                });
                _tree = BVH<device_memory_space>(device_execution_space(), cloud);
            }
        };
    
        ~PointCloudSearch() {};

        //! Returns a liberal estimated upper bound on number of neighbors to be returned by a neighbor search
        //! for a given choice of dimension, basis size, and epsilon_multiplier. Assumes quasiuniform distribution
        //! of points. This result can be used to size a preallocated neighbor_lists kokkos view.
        static inline int getEstimatedNumberNeighborsUpperBound(int unisolvency_size, const int dimension, const double epsilon_multiplier) {
            int multiplier = 1;
            if (dimension==1) multiplier = 2;
            return multiplier * 2.0 * unisolvency_size * std::pow(epsilon_multiplier, dimension) + 1; // +1 is for the number of neighbors entry needed in the first entry of each row
        }

        //! Returns the distance between a point and a source site, given its index
        inline double kdtreeDistance(const double* queryPt, const int idx) const {

            double distance = 0;
            for (int i=0; i<_dim; ++i) {
                distance += (_src_pts_view(idx,i)-queryPt[i])*(_src_pts_view(idx,i)-queryPt[i]);
            }
            return std::sqrt(distance);

        }

        /*! \brief Generates neighbor lists of 2D view by performing a radius search 
            where the radius to be searched is in the epsilons view.
            If uniform_radius is given, then this overrides the epsilons view radii sizes.
            Accepts 2D neighbor_lists without number_of_neighbors_list.
            \param is_dry_run               [in] - whether to do a dry-run (find neighbors, but don't store)
            \param trg_pts_view             [in] - target coordinates from which to seek neighbors
            \param neighbor_lists           [out] - 2D view of neighbor lists to be populated from search
            \param epsilons                 [in/out] - radius to search, overwritten if uniform_radius != 0
            \param uniform_radius           [in] - double != 0 determines whether to overwrite all epsilons for uniform search
            \param max_search_radius        [in] - largest valid search (useful only for MPI jobs if halo size exists)
        */
        template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
        size_t generate2DNeighborListsFromRadiusSearch(bool is_dry_run, trg_view_type trg_pts_view, 
                neighbor_lists_view_type& neighbor_lists, epsilons_view_type epsilons, 
                const double uniform_radius = 0.0, double max_search_radius = 0.0) {

            // function does not populate epsilons, they must be prepopulated
            const int num_target_sites = trg_pts_view.extent(0);
 
            // check neighbor lists and epsilons view sizes
            compadre_assert_release((neighbor_lists.extent(0)==(size_t)num_target_sites 
                        && neighbor_lists.extent(1)>=1)
                        && "neighbor lists View does not have large enough dimensions");
            compadre_assert_release((neighbor_lists_view_type::rank==2) && "neighbor_lists must be a 2D Kokkos view.");


            // temporary 1D array for cr_neighbor_lists
            typedef typename neighbor_lists_view_type::value_type value_type;
            size_t cr_neighbor_entries =  (is_dry_run) ? 0 : num_target_sites * neighbor_lists.extent(1);
            Kokkos::View<value_type*, typename neighbor_lists_view_type::memory_space> 
                cr_neighbor_lists(
                    Kokkos::view_alloc(neighbor_lists.label(), 
                        typename neighbor_lists_view_type::memory_space()), 
                    cr_neighbor_entries);

            // temporary 1D array for number_of_neighbors_list
            typedef typename neighbor_lists_view_type::value_type value_type;
            Kokkos::View<value_type*, typename neighbor_lists_view_type::memory_space> 
                number_of_neighbors_list(
                    Kokkos::view_alloc("number of neighbors list", 
                        typename neighbor_lists_view_type::memory_space()), 
                    num_target_sites);

            if (!is_dry_run) { 
                // number_of_neighbors_list not populated when !is_dry_run
                Kokkos::parallel_for(num_target_sites, KOKKOS_LAMBDA(int i) {
                    number_of_neighbors_list(i) = neighbor_lists(i,0);
                });
            }
            
            // doesn't fill number_of_neighbors_list when !is_dry_run
            generateCRNeighborListsFromRadiusSearch(is_dry_run, trg_pts_view,
                cr_neighbor_lists, number_of_neighbors_list, epsilons, uniform_radius, max_search_radius);

            size_t max_num_neighbors = 0;
            if (!is_dry_run) {
                auto nla = CreateNeighborLists(cr_neighbor_lists, number_of_neighbors_list);
                ConvertCompressedRowNeighborListsTo2D(nla, neighbor_lists);
                max_num_neighbors = nla.getMaxNumNeighbors();
            } else {
                auto nla = CreateNeighborLists(number_of_neighbors_list);
                max_num_neighbors = nla.getMaxNumNeighbors();
            }

            return max_num_neighbors;
        }

        /*! \brief Generates compressed row neighbor lists by performing a radius search 
            where the radius to be searched is in the epsilons view.
            If uniform_radius is given, then this overrides the epsilons view radii sizes.
            Accepts 1D neighbor_lists with 1D number_of_neighbors_list.
            \param is_dry_run               [in] - whether to do a dry-run (find neighbors, but don't store)
            \param trg_pts_view             [in] - target coordinates from which to seek neighbors
            \param neighbor_lists           [out] - 1D view of neighbor lists to be populated from search
            \param number_of_neighbors_list [in/out] - number of neighbors for each target site
            \param epsilons                 [in/out] - radius to search, overwritten if uniform_radius != 0
            \param uniform_radius           [in] - double != 0 determines whether to overwrite all epsilons for uniform search
            \param max_search_radius        [in] - largest valid search (useful only for MPI jobs if halo size exists)
        */
        template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
        size_t generateCRNeighborListsFromRadiusSearch(bool is_dry_run, trg_view_type trg_pts_view, 
                neighbor_lists_view_type neighbor_lists, neighbor_lists_view_type number_of_neighbors_list, 
                epsilons_view_type epsilons, const double uniform_radius = 0.0, double max_search_radius = 0.0) {

            // function does not populate epsilons, they must be prepopulated

            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename trg_view_type::memory_space>::accessible==1) &&
                    "Target coordinates view passed to generateCRNeighborListsFromRadiusSearch should be accessible from the host.");
            compadre_assert_release((((int)trg_pts_view.extent(1))>=_dim) &&
                    "Target coordinates view passed to generateCRNeighborListsFromRadiusSearch must have \
                    second dimension as large as _dim.");
            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename neighbor_lists_view_type::memory_space>::accessible==1) &&
                    "Views passed to generateCRNeighborListsFromRadiusSearch should be accessible from the host.");
            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename epsilons_view_type::memory_space>::accessible==1) &&
                    "Views passed to generateCRNeighborListsFromRadiusSearch should be accessible from the host.");

            // loop size
            const int num_target_sites = trg_pts_view.extent(0);

            compadre_assert_release((number_of_neighbors_list.extent(0)==(size_t)num_target_sites)
                        && "number_of_neighbors_list or neighbor lists View does not have large enough dimensions");
            compadre_assert_release((neighbor_lists_view_type::rank==1) && "neighbor_lists must be a 1D Kokkos view.");

            compadre_assert_release((epsilons.extent(0)==(size_t)num_target_sites)
                        && "epsilons View does not have the correct dimension");

            typedef Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
                    scratch_double_view;

            // build target points into a cloud of sphere with radius given as epsilons(i)
            Kokkos::View<ArborX::Sphere *> cloud("trg_point_cloud", trg_pts_view.extent(0));
            Kokkos::parallel_for(num_target_sites, KOKKOS_LAMBDA(int i) {
                compadre_kernel_assert_release((epsilons(i)<=max_search_radius || max_search_radius==0) && "max_search_radius given (generally derived from the size of a halo region), and search radius needed would exceed this max_search_radius.");
                if (uniform_radius > 0) epsilons(i) = uniform_radius;
                if (_dim==1) {
                    cloud(i) = {{{trg_pts_view(i,0),0,0}}, epsilons(i)};
                }
                if (_dim==2) {
                    cloud(i) = {{{trg_pts_view(i,0),trg_pts_view(i,1),0}}, epsilons(i)};
                }
                if (_dim==3) {
                    cloud(i) = {{{trg_pts_view(i,0),trg_pts_view(i,1),trg_pts_view(i,2)}}, epsilons(i)};
                }
            });
            Kokkos::fence();

            // build queries
            Kokkos::View<ArborX::Intersects<ArborX::Sphere> *, device_memory_space> 
                queries("queries", num_target_sites);
            Kokkos::parallel_for(Kokkos::RangePolicy<device_execution_space>(0, num_target_sites),
            KOKKOS_LAMBDA(int i) {
                queries(i) = ArborX::intersects(cloud(i));
            });
            Kokkos::fence();

            // perform tree search
            Kokkos::View<int *, device_memory_space> values("values", 0);
            Kokkos::View<int *, device_memory_space> offsets("offsets", 0);
            _tree.query(device_execution_space(), queries, values, offsets);


            // set number of neighbors list (how many neighbors for each target site) based
            // on results on tree query
            Kokkos::parallel_for(Kokkos::RangePolicy<device_execution_space>(0, num_target_sites),
            KOKKOS_LAMBDA(int i) {
                size_t neighbors_found_for_target_i = 0;
                if (i==num_target_sites-1) {
                    neighbors_found_for_target_i = values.extent(0) - offsets(i);
                } else {
                    neighbors_found_for_target_i = offsets(i+1) - offsets(i);
                }
                if (!is_dry_run) { // check it is as expected before overwriting
                    compadre_kernel_assert_debug((neighbors_found==(size_t)number_of_neighbors_list(i)) 
                            && "Number of neighbors found changed since dry-run.");
                }
                number_of_neighbors_list(i) = neighbors_found_for_target_i;
            });
            Kokkos::fence();

            // build a row offsets object (only if not a dry run)
            typedef Kokkos::View<global_index_type*, typename neighbor_lists_view_type::array_layout,
                    typename neighbor_lists_view_type::memory_space, typename neighbor_lists_view_type::memory_traits> row_offsets_view_type;
            row_offsets_view_type row_offsets;
            int max_neighbor_list_row_storage_size = 1;
            if (!is_dry_run) {
                auto nla = CreateNeighborLists(neighbor_lists, number_of_neighbors_list);
                max_neighbor_list_row_storage_size = nla.getMaxNumNeighbors();
                Kokkos::resize(row_offsets, num_target_sites);
                Kokkos::fence();
                Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0,num_target_sites), [&](const int i) {
                    row_offsets(i) = nla.getRowOffsetHost(i); 
                });
                Kokkos::fence();
            }

            // sort to put closest neighbor first (other neighbors are not affected)
            int team_scratch_size = 0;
            team_scratch_size += scratch_double_view::shmem_size(trg_pts_view.extent(1)); // distances
            Kokkos::parallel_for("radius search", host_team_policy(num_target_sites, Kokkos::AUTO)
                    .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
                    KOKKOS_LAMBDA(const host_member_type& teamMember) {

                const int i = teamMember.league_rank();

                // make unmanaged scratch views
                scratch_double_view trg_pt(teamMember.team_scratch(0 /*shared memory*/), trg_pts_view.extent(1));
                for (int j=0; j<trg_pts_view.extent(1); ++j) {
                    trg_pt(j) = trg_pts_view(i,j);
                }

                // find swap index
                int min_distance_ind = 0;
                double first_neighbor_distance = kdtreeDistance(trg_pt.data(), values(offsets(i)));
                for (int j=1; j<number_of_neighbors_list(i); ++j) {
                    // distance to neighbor
                    double neighbor_distance = kdtreeDistance(trg_pt.data(), values(offsets(i)+j));
                    if (neighbor_distance < first_neighbor_distance) {
                        first_neighbor_distance = neighbor_distance;
                        min_distance_ind = j;
                    }
                }
                // do the swap
                if (min_distance_ind != 0) {
                    int tmp = values(offsets(i));
                    values(offsets(i)) = values(offsets(i)+min_distance_ind);
                    values(offsets(i)+min_distance_ind) = tmp;

                }
            });
            Kokkos::fence();

            // copy neighbor list values over from tree query
            if (!is_dry_run) {
                Kokkos::parallel_for(Kokkos::RangePolicy<device_execution_space>(0, num_target_sites),
                KOKKOS_LAMBDA(int i) {
                    for (int j=0; j<number_of_neighbors_list(i); ++j) {
                        neighbor_lists(row_offsets(i)+j) = values(offsets(i)+j);
                    }
                });
                Kokkos::fence();
            }

            auto nla = CreateNeighborLists(number_of_neighbors_list);
            return nla.getTotalNeighborsOverAllListsHost();
        }


        /*! \brief Generates neighbor lists as 2D view by performing a k-nearest neighbor search
            Only accepts 2D neighbor_lists without number_of_neighbors_list.
            \param is_dry_run               [in] - whether to do a dry-run (find neighbors, but don't store)
            \param trg_pts_view             [in] - target coordinates from which to seek neighbors
            \param neighbor_lists           [out] - 2D view of neighbor lists to be populated from search
            \param epsilons                 [in/out] - radius to search, overwritten if uniform_radius != 0
            \param neighbors_needed         [in] - k neighbors needed as a minimum
            \param epsilon_multiplier       [in] - distance to kth neighbor multiplied by epsilon_multiplier for follow-on radius search
            \param max_search_radius        [in] - largest valid search (useful only for MPI jobs if halo size exists)
        */
        template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
        size_t generate2DNeighborListsFromKNNSearch(bool is_dry_run, trg_view_type trg_pts_view, 
                neighbor_lists_view_type neighbor_lists, epsilons_view_type epsilons, 
                const int neighbors_needed, const double epsilon_multiplier = 1.6, 
                double max_search_radius = 0.0) {

            // First, do a knn search (removes need for guessing initial search radius)

            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename trg_view_type::memory_space>::accessible==1) &&
                    "Target coordinates view passed to generate2DNeighborListsFromKNNSearch should be accessible from the host.");
            compadre_assert_release((((int)trg_pts_view.extent(1))>=_dim) &&
                    "Target coordinates view passed to generate2DNeighborListsFromRadiusSearch must have \
                    second dimension as large as _dim.");
            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename neighbor_lists_view_type::memory_space>::accessible==1) &&
                    "Views passed to generate2DNeighborListsFromKNNSearch should be accessible from the host.");
            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename epsilons_view_type::memory_space>::accessible==1) &&
                    "Views passed to generate2DNeighborListsFromKNNSearch should be accessible from the host.");

            // loop size
            const int num_target_sites = trg_pts_view.extent(0);

            //if ((!_tree_1d && _dim==1) || (!_tree_2d && _dim==2) || (!_tree_3d && _dim==3)) {
            //    this->generateKDTree();
            //}
            //Kokkos::fence();

            //compadre_assert_release((num_target_sites==0 || // sizes don't matter when there are no targets
            //        (neighbor_lists.extent(0)==(size_t)num_target_sites 
            //            && neighbor_lists.extent(1)>=(size_t)(neighbors_needed+1)))
            //            && "neighbor lists View does not have large enough dimensions");
            //compadre_assert_release((neighbor_lists_view_type::rank==2) && "neighbor_lists must be a 2D Kokkos view.");

            //compadre_assert_release((epsilons.extent(0)==(size_t)num_target_sites)
            //            && "epsilons View does not have the correct dimension");

            //typedef Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
            //        scratch_double_view;

            //typedef Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
            //        scratch_int_view;

            //// determine scratch space size needed
            //int team_scratch_size = 0;
            //team_scratch_size += scratch_double_view::shmem_size(neighbor_lists.extent(1)); // distances
            //team_scratch_size += scratch_int_view::shmem_size(neighbor_lists.extent(1)); // indices
            //team_scratch_size += scratch_double_view::shmem_size(_dim); // target coordinate

            //// minimum number of neighbors found over all target sites' neighborhoods
            //size_t min_num_neighbors = 0;
            ////
            //// part 1. do knn search for neighbors needed for unisolvency
            //// each row of neighbor lists is a neighbor list for the target site corresponding to that row
            ////
            //// as long as neighbor_lists can hold the number of neighbors_needed, we don't need to check
            //// that the maximum number of neighbors will fit into neighbor_lists
            ////
            //Kokkos::parallel_reduce("knn search", host_team_policy(num_target_sites, Kokkos::AUTO)
            //        .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
            //        KOKKOS_LAMBDA(const host_member_type& teamMember, size_t& t_min_num_neighbors) {

            //    // make unmanaged scratch views
            //    scratch_double_view neighbor_distances(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.extent(1));
            //    scratch_int_view neighbor_indices(teamMember.team_scratch(0 /*shared memory*/), neighbor_lists.extent(1));
            //    scratch_double_view this_target_coord(teamMember.team_scratch(0 /*shared memory*/), _dim);

            //    size_t neighbors_found = 0;

            //    const int i = teamMember.league_rank();

            //    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, neighbor_lists.extent(1)), [=](const int j) {
            //        neighbor_indices(j) = 0;
            //        neighbor_distances(j) = -1.0;
            //    });
            //
            //    teamMember.team_barrier();
            //    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
            //        // target_coords is LayoutLeft on device and its HostMirror, so giving a pointer to 
            //        // this data would lead to a wrong result if the device is a GPU

            //        for (int j=0; j<_dim; ++j) { 
            //            this_target_coord(j) = trg_pts_view(i,j);
            //        }

            //        if (_dim==1) {
            //            neighbors_found = _tree_1d->knnSearch(this_target_coord.data(), neighbors_needed, 
            //                    neighbor_indices.data(), neighbor_distances.data()) ;
            //        } else if (_dim==2) {
            //            neighbors_found = _tree_2d->knnSearch(this_target_coord.data(), neighbors_needed, 
            //                    neighbor_indices.data(), neighbor_distances.data()) ;
            //        } else if (_dim==3) {
            //            neighbors_found = _tree_3d->knnSearch(this_target_coord.data(), neighbors_needed, 
            //                    neighbor_indices.data(), neighbor_distances.data()) ;
            //        }

            //        // get minimum number of neighbors found over all target sites' neighborhoods
            //        t_min_num_neighbors = (neighbors_found < t_min_num_neighbors) ? neighbors_found : t_min_num_neighbors;
            //
            //        // scale by epsilon_multiplier to window from location where the last neighbor was found
            //        epsilons(i) = (neighbor_distances(neighbors_found-1) > 0) ?
            //            std::sqrt(neighbor_distances(neighbors_found-1))*epsilon_multiplier : 1e-14*epsilon_multiplier;
            //        // the only time the second case using 1e-14 is used is when either zero neighbors or exactly one 
            //        // neighbor (neighbor is target site) is found.  when the follow on radius search is conducted, the one
            //        // neighbor (target site) will not be found if left at 0, so any positive amount will do, however 1e-14 
            //        // should is small enough to ensure that other neighbors are not found

            //        // needs furthest neighbor's distance for next portion
            //        compadre_kernel_assert_release((neighbors_found<neighbor_lists.extent(1) || is_dry_run) 
            //                && "Neighbors found exceeded storage capacity in neighbor list.");
            //        compadre_kernel_assert_release((epsilons(i)<=max_search_radius || max_search_radius==0 || is_dry_run) 
            //                && "max_search_radius given (generally derived from the size of a halo region), \
            //                    and search radius needed would exceed this max_search_radius.");
            //        // neighbor_distances stores squared distances from neighbor to target, as returned by nanoflann
            //    });
            //}, Kokkos::Min<size_t>(min_num_neighbors) );
            //Kokkos::fence();
            //
            //// if no target sites, then min_num_neighbors is set to neighbors_needed
            //// which also avoids min_num_neighbors being improperly set by min reduction
            //if (num_target_sites==0) min_num_neighbors = neighbors_needed;

            //// Next, check that we found the neighbors_needed number that we require for unisolvency
            //compadre_assert_release((num_target_sites==0 || (min_num_neighbors>=(size_t)neighbors_needed))
            //        && "Neighbor search failed to find number of neighbors needed for unisolvency.");
            //
            //// call a radius search using values now stored in epsilons
            //size_t max_num_neighbors = generate2DNeighborListsFromRadiusSearch(is_dry_run, trg_pts_view, neighbor_lists, 
            //        epsilons, 0.0 /*don't set uniform radius*/, max_search_radius);

            return 0;//max_num_neighbors;
        }

        /*! \brief Generates compressed row neighbor lists by performing a k-nearest neighbor search
            Only accepts 1D neighbor_lists with 1D number_of_neighbors_list.
            \param is_dry_run               [in] - whether to do a dry-run (find neighbors, but don't store)
            \param trg_pts_view             [in] - target coordinates from which to seek neighbors
            \param neighbor_lists           [out] - 1D view of neighbor lists to be populated from search
            \param number_of_neighbors_list [in/out] - number of neighbors for each target site
            \param epsilons                 [in/out] - radius to search, overwritten if uniform_radius != 0
            \param neighbors_needed         [in] - k neighbors needed as a minimum
            \param epsilon_multiplier       [in] - distance to kth neighbor multiplied by epsilon_multiplier for follow-on radius search
            \param max_search_radius        [in] - largest valid search (useful only for MPI jobs if halo size exists)
        */
        template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
        size_t generateCRNeighborListsFromKNNSearch(bool is_dry_run, trg_view_type trg_pts_view, 
                neighbor_lists_view_type neighbor_lists, neighbor_lists_view_type number_of_neighbors_list,
                epsilons_view_type epsilons, const int neighbors_needed, const double epsilon_multiplier = 1.6, 
                double max_search_radius = 0.0) {

            // First, do a knn search (removes need for guessing initial search radius)

            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename trg_view_type::memory_space>::accessible==1) &&
                    "Target coordinates view passed to generateCRNeighborListsFromKNNSearch should be accessible from the host.");
            compadre_assert_release((((int)trg_pts_view.extent(1))>=_dim) &&
                    "Target coordinates view passed to generateCRNeighborListsFromRadiusSearch must have \
                    second dimension as large as _dim.");
            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename neighbor_lists_view_type::memory_space>::accessible==1) &&
                    "Views passed to generateCRNeighborListsFromKNNSearch should be accessible from the host.");
            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename epsilons_view_type::memory_space>::accessible==1) &&
                    "Views passed to generateCRNeighborListsFromKNNSearch should be accessible from the host.");

            // loop size
            const int num_target_sites = trg_pts_view.extent(0);

            //if ((!_tree_1d && _dim==1) || (!_tree_2d && _dim==2) || (!_tree_3d && _dim==3)) {
            //    this->generateKDTree();
            //}
            //Kokkos::fence();

            //compadre_assert_release((number_of_neighbors_list.extent(0)==(size_t)num_target_sites ) 
            //            && "number_of_neighbors_list or neighbor lists View does not have large enough dimensions");
            //compadre_assert_release((neighbor_lists_view_type::rank==1) && "neighbor_lists must be a 1D Kokkos view.");

            //// if dry-run, neighbors_needed, else max over previous dry-run
            //int max_neighbor_list_row_storage_size = neighbors_needed;
            //if (!is_dry_run) {
            //    auto nla = CreateNeighborLists(neighbor_lists, number_of_neighbors_list);
            //    max_neighbor_list_row_storage_size = nla.getMaxNumNeighbors();
            //}

            //compadre_assert_release((epsilons.extent(0)==(size_t)num_target_sites)
            //            && "epsilons View does not have the correct dimension");

            //typedef Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
            //        scratch_double_view;

            //typedef Kokkos::View<size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
            //        scratch_int_view;

            //// determine scratch space size needed
            //int team_scratch_size = 0;
            //team_scratch_size += scratch_double_view::shmem_size(max_neighbor_list_row_storage_size); // distances
            //team_scratch_size += scratch_int_view::shmem_size(max_neighbor_list_row_storage_size); // indices
            //team_scratch_size += scratch_double_view::shmem_size(_dim); // target coordinate

            //// minimum number of neighbors found over all target sites' neighborhoods
            //size_t min_num_neighbors = 0;
            ////
            //// part 1. do knn search for neighbors needed for unisolvency
            //// each row of neighbor lists is a neighbor list for the target site corresponding to that row
            ////
            //// as long as neighbor_lists can hold the number of neighbors_needed, we don't need to check
            //// that the maximum number of neighbors will fit into neighbor_lists
            ////
            //Kokkos::parallel_reduce("knn search", host_team_policy(num_target_sites, Kokkos::AUTO)
            //        .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
            //        KOKKOS_LAMBDA(const host_member_type& teamMember, size_t& t_min_num_neighbors) {

            //    // make unmanaged scratch views
            //    scratch_double_view neighbor_distances(teamMember.team_scratch(0 /*shared memory*/), max_neighbor_list_row_storage_size);
            //    scratch_int_view neighbor_indices(teamMember.team_scratch(0 /*shared memory*/), max_neighbor_list_row_storage_size);
            //    scratch_double_view this_target_coord(teamMember.team_scratch(0 /*shared memory*/), _dim);

            //    size_t neighbors_found = 0;

            //    const int i = teamMember.league_rank();

            //    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, max_neighbor_list_row_storage_size), [=](const int j) {
            //        neighbor_indices(j) = 0;
            //        neighbor_distances(j) = -1.0;
            //    });
            //
            //    teamMember.team_barrier();
            //    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
            //        // target_coords is LayoutLeft on device and its HostMirror, so giving a pointer to 
            //        // this data would lead to a wrong result if the device is a GPU

            //        for (int j=0; j<_dim; ++j) { 
            //            this_target_coord(j) = trg_pts_view(i,j);
            //        }

            //        if (_dim==1) {
            //            neighbors_found = _tree_1d->knnSearch(this_target_coord.data(), neighbors_needed, 
            //                    neighbor_indices.data(), neighbor_distances.data()) ;
            //        } else if (_dim==2) {
            //            neighbors_found = _tree_2d->knnSearch(this_target_coord.data(), neighbors_needed, 
            //                    neighbor_indices.data(), neighbor_distances.data()) ;
            //        } else if (_dim==3) {
            //            neighbors_found = _tree_3d->knnSearch(this_target_coord.data(), neighbors_needed, 
            //                    neighbor_indices.data(), neighbor_distances.data()) ;
            //        }

            //        // get minimum number of neighbors found over all target sites' neighborhoods
            //        t_min_num_neighbors = (neighbors_found < t_min_num_neighbors) ? neighbors_found : t_min_num_neighbors;
            //
            //        // scale by epsilon_multiplier to window from location where the last neighbor was found
            //        epsilons(i) = (neighbor_distances(neighbors_found-1) > 0) ?
            //            std::sqrt(neighbor_distances(neighbors_found-1))*epsilon_multiplier : 1e-14*epsilon_multiplier;
            //        // the only time the second case using 1e-14 is used is when either zero neighbors or exactly one 
            //        // neighbor (neighbor is target site) is found.  when the follow on radius search is conducted, the one
            //        // neighbor (target site) will not be found if left at 0, so any positive amount will do, however 1e-14 
            //        // should is small enough to ensure that other neighbors are not found

            //        compadre_kernel_assert_release((epsilons(i)<=max_search_radius || max_search_radius==0 || is_dry_run) 
            //                && "max_search_radius given (generally derived from the size of a halo region), \
            //                    and search radius needed would exceed this max_search_radius.");
            //        // neighbor_distances stores squared distances from neighbor to target, as returned by nanoflann
            //    });
            //}, Kokkos::Min<size_t>(min_num_neighbors) );
            //Kokkos::fence();
            //
            //// if no target sites, then min_num_neighbors is set to neighbors_needed
            //// which also avoids min_num_neighbors being improperly set by min reduction
            //if (num_target_sites==0) min_num_neighbors = neighbors_needed;

            //// Next, check that we found the neighbors_needed number that we require for unisolvency
            //compadre_assert_release((num_target_sites==0 || (min_num_neighbors>=(size_t)neighbors_needed))
            //        && "Neighbor search failed to find number of neighbors needed for unisolvency.");
            //
            //// call a radius search using values now stored in epsilons
            //generateCRNeighborListsFromRadiusSearch(is_dry_run, trg_pts_view, neighbor_lists, 
            //        number_of_neighbors_list, epsilons, 0.0 /*don't set uniform radius*/, max_search_radius);

            auto nla = CreateNeighborLists(number_of_neighbors_list);
            return nla.getTotalNeighborsOverAllListsHost();
        }
}; // PointCloudSearch

//! CreatePointCloudSearch allows for the construction of an object of type PointCloudSearch with template deduction
template <typename view_type>
PointCloudSearch<view_type> CreatePointCloudSearch(view_type src_view, const local_index_type dimensions = -1, const local_index_type max_leaf = -1) { 
    return PointCloudSearch<view_type>(src_view, dimensions, max_leaf);
}

} // Compadre

#endif
