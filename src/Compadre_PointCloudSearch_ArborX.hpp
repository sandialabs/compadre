#ifndef _COMPADRE_POINTCLOUDSEARCHARBORX_HPP_
#define _COMPADRE_POINTCLOUDSEARCHARBORX_HPP_

#include "Compadre_Typedefs.hpp"
#include "Compadre_NeighborLists.hpp"
#include <Kokkos_Core.hpp>
#include <ArborX_LinearBVH.hpp>

template <typename view_type_1>
struct Points {
    typedef decltype(Kokkos::create_mirror_view<device_memory_space>(
                device_memory_space(), view_type_1())) 
                        device_mirror_type;
    device_mirror_type _pts;

    Points(view_type_1 pts) {
        _pts = Kokkos::create_mirror_view<device_memory_space>(device_memory_space(), pts);
        Kokkos::deep_copy(_pts, pts);
        Kokkos::fence();
    }
};

template <typename view_type>
struct ArborX::AccessTraits<Points<view_type>, ArborX::PrimitivesTag>
{
    static KOKKOS_FUNCTION std::size_t size(Points<view_type> const &cloud)
    {
        return cloud._pts.extent(0);
    }
    static KOKKOS_FUNCTION ArborX::Point get(Points<view_type> const &cloud, std::size_t i)
    {
        switch (cloud._pts.extent(1)) {
        case 3:
            return ArborX::Point(cloud._pts(i,0), cloud._pts(i,1), cloud._pts(i,2));
        case 2:
            return ArborX::Point(cloud._pts(i,0), cloud._pts(i,1),             0.0);
        case 1:
            return ArborX::Point(cloud._pts(i,0),             0.0,             0.0);
        default:
            compadre_kernel_assert_release(false && "Invalid dimension for cloud.");
            return ArborX::Point(           0.0,             0.0,             0.0);
        }
    }
    using memory_space = device_memory_space;
};

template <typename view_type_1, typename view_type_2>
struct Radius {

    typedef decltype(Kokkos::create_mirror_view<device_memory_space>(
                device_memory_space(), view_type_1())) 
                        device_mirror_type_1;
    device_mirror_type_1 _pts;
    typedef decltype(Kokkos::create_mirror_view<device_memory_space>(
                device_memory_space(), view_type_2())) 
                        device_mirror_type_2;
    device_mirror_type_2 _radii;
    double _uniform_radius;
    double _max_search_radius;

    Radius(view_type_1 pts, view_type_2 radii, const double uniform_radius, 
            const double max_search_radius) :
                _uniform_radius(uniform_radius), 
                _max_search_radius(max_search_radius) {
        _pts = Kokkos::create_mirror_view<device_memory_space>(device_memory_space(), pts);
        _radii = Kokkos::create_mirror_view<device_memory_space>(device_memory_space(), radii);
        Kokkos::deep_copy(_pts, pts);
        Kokkos::deep_copy(_radii, radii);
        Kokkos::fence();
    }

    void getRadiiFromDevice(view_type_2 radii) {
        Kokkos::deep_copy(radii, _radii);
        Kokkos::fence();
    }
};

template <typename view_type_1, typename view_type_2>
struct ArborX::AccessTraits<Radius<view_type_1, view_type_2>, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(Radius<view_type_1, view_type_2> const &cloud)
  {
      return cloud._pts.extent(0);
  }
  static KOKKOS_FUNCTION ArborX::Intersects<ArborX::Sphere> 
      get(Radius<view_type_1, view_type_2> const &cloud, std::size_t i)
  {
      search_scalar radius = (cloud._uniform_radius != 0) ? cloud._uniform_radius : cloud._radii(i);
      cloud._radii(i) = radius;
      compadre_kernel_assert_release((cloud._radii(i)<=cloud._max_search_radius || cloud._max_search_radius==0) 
              && "max_search_radius given (generally derived from the size of a halo region), \
              and search radius needed would exceed this max_search_radius.");
      switch (cloud._pts.extent(1)) {
      case 3:
          return ArborX::intersects(ArborX::Sphere(
                      ArborX::Point(cloud._pts(i,0), cloud._pts(i,1), cloud._pts(i,2)), 
                      radius));
      case 2:
          return ArborX::intersects(ArborX::Sphere(
                      ArborX::Point(cloud._pts(i,0), cloud._pts(i,1),             0.0), 
                      radius));
      case 1:
          return ArborX::intersects(ArborX::Sphere(
                      ArborX::Point(cloud._pts(i,0),             0.0,             0.0), 
                      radius));
      default:
          compadre_kernel_assert_release(false && "Invalid dimension for cloud.");
          return ArborX::intersects(ArborX::Sphere(
                      ArborX::Point(            0.0,             0.0,             0.0),
                      radius));
      }
  }
  using memory_space = device_memory_space;
};

template <typename view_type_1>
struct NearestNeighbor {
    typedef decltype(Kokkos::create_mirror_view<device_memory_space>(
                device_memory_space(), view_type_1())) 
                        device_mirror_type;
    device_mirror_type _pts;
    int _neighbors_needed;

    NearestNeighbor(view_type_1 pts, const int neighbors_needed) :
            _neighbors_needed(neighbors_needed) {
        _pts = Kokkos::create_mirror_view<device_memory_space>(device_memory_space(), pts);
        Kokkos::deep_copy(_pts, pts);
        Kokkos::fence();
    }
};

template <typename view_type_1>
struct ArborX::AccessTraits<NearestNeighbor<view_type_1>, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(NearestNeighbor<view_type_1> const &cloud)
  {
      return cloud._pts.extent(0);
  }
  static KOKKOS_FUNCTION ArborX::Nearest<ArborX::Point> 
      get(NearestNeighbor<view_type_1> const &cloud, std::size_t i)
  {
      switch (cloud._pts.extent(1)) {
      case 3:
          return ArborX::nearest(
                      ArborX::Point(cloud._pts(i,0), cloud._pts(i,1), cloud._pts(i,2)), 
                      cloud._neighbors_needed);
      case 2:
          return ArborX::nearest(
                      ArborX::Point(cloud._pts(i,0), cloud._pts(i,1),             0.0), 
                      cloud._neighbors_needed);
      case 1:
          return ArborX::nearest(
                      ArborX::Point(cloud._pts(i,0),             0.0,             0.0), 
                      cloud._neighbors_needed);
      default:
          compadre_kernel_assert_release(false && "Invalid dimension for cloud.");
          return ArborX::nearest(
                      ArborX::Point(            0.0,             0.0,             0.0),
                      cloud._neighbors_needed);
      }
  }
  using memory_space = device_memory_space;
};

namespace Compadre {

//!  PointCloudSearch generates neighbor lists and window sizes for each target site, using ArborX
/*!
*
*    If a 2D view for `neighbors_list` is used, then \f$ N(i,j+1) \f$ will store the \f$ j^{th} \f$ neighbor of \f$ i \f$,
*    and \f$ N(i,0) \f$ will store the number of neighbors for target \f$ i \f$.
*
*    Otherwise, a compressed row representation will be stored in neighbor_lists and the number of
*    neighbors for each target site will be stored in number_of_neighbors_list.
*
*/
template <typename view_type>
class PointCloudSearch {

    protected:

        //! source site coordinates
        view_type _src_pts_view;
        local_index_type _dim;
        local_index_type _max_leaf;

        ArborX::BVH<device_memory_space> _tree;

    public:

        PointCloudSearch(view_type src_pts_view, const local_index_type dimension = -1,
                const local_index_type max_leaf = -1) 
                : _src_pts_view(src_pts_view), 
                  _dim((dimension < 0) ? src_pts_view.extent(1) : dimension),
                  _max_leaf((max_leaf < 0) ? 10 : max_leaf) {
            compadre_assert_release((Kokkos::SpaceAccessibility<host_execution_space, typename view_type::memory_space>::accessible==1)
                    && "Views passed to PointCloudSearch at construction should be accessible from the host.");

            {
                _tree = ArborX::BVH<device_memory_space>(device_execution_space(), 
                                Points<view_type>(_src_pts_view));
            }
        };
    
        ~PointCloudSearch() {};

        //! Returns the distance between a point and a source site, given its index
        inline search_scalar kdtreeDistance(const double* queryPt, const int idx) const {
            search_scalar distance = 0;
            for (int i=0; i<_dim; ++i) {
                distance += (static_cast<search_scalar>(_src_pts_view(idx,i))
                                -static_cast<search_scalar>(queryPt[i]))
                           *(static_cast<search_scalar>(_src_pts_view(idx,i))
                                 -static_cast<search_scalar>(queryPt[i]));
            }
            return std::sqrt(distance);
        }

        /*! \brief Generates neighbor lists of 2D view by performing a radius search 
            where the radius to be searched is in the epsilons view.
            If uniform_radius is given, then this overrides the epsilons view radii sizes.
            Accepts 2D neighbor_lists without number_of_neighbors_list.
            \param trg_pts_view             [in] - target coordinates from which to seek neighbors
            \param neighbor_lists           [out] - 2D view of neighbor lists to be populated from search
            \param epsilons                 [in/out] - radius to search, overwritten if uniform_radius != 0
            \param uniform_radius           [in] - double != 0 determines whether to overwrite all epsilons for uniform search
            \param max_search_radius        [in] - largest valid search (useful only for MPI jobs if halo size exists)
            \param check_same               [in] - ignored (to match nanoflann)
            \param do_dry_run               [in] - ignored (to match nanoflann)
        */
        template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
        size_t generate2DNeighborListsFromRadiusSearch(trg_view_type trg_pts_view, 
                neighbor_lists_view_type& neighbor_lists, epsilons_view_type& epsilons, 
                const double uniform_radius = 0.0, double max_search_radius = 0.0,
                bool check_same = true, bool do_dry_run = true) {

            const int num_target_sites = trg_pts_view.extent(0);
 
            // check neighbor lists rank
            compadre_assert_release((neighbor_lists_view_type::rank==2) && "neighbor_lists must be a 2D Kokkos view.");

            // temporary 1D array for cr_neighbor_lists
            typedef typename neighbor_lists_view_type::value_type value_type;
            Kokkos::View<value_type*, typename neighbor_lists_view_type::memory_space> 
                cr_neighbor_lists(neighbor_lists.label(), 0);

            // temporary 1D array for number_of_neighbors_list
            typedef typename neighbor_lists_view_type::value_type value_type;
            Kokkos::View<value_type*, typename neighbor_lists_view_type::memory_space> 
                number_of_neighbors_list("number of neighbors list", num_target_sites);

            generateCRNeighborListsFromRadiusSearch(trg_pts_view,
                cr_neighbor_lists, number_of_neighbors_list, epsilons, uniform_radius, 
                max_search_radius, check_same, do_dry_run);

            auto nla = CreateNeighborLists(cr_neighbor_lists, number_of_neighbors_list);
            size_t max_num_neighbors = nla.getMaxNumNeighbors();
            ConvertCompressedRowNeighborListsTo2D(nla, neighbor_lists);
            
            return max_num_neighbors;
        }

        /*! \brief Generates compressed row neighbor lists by performing a radius search 
            where the radius to be searched is in the epsilons view.
            If uniform_radius is given, then this overrides the epsilons view radii sizes.
            Accepts 1D neighbor_lists with 1D number_of_neighbors_list.
            \param trg_pts_view             [in] - target coordinates from which to seek neighbors
            \param neighbor_lists           [out] - 1D view of neighbor lists to be populated from search
            \param number_of_neighbors_list [in/out] - number of neighbors for each target site
            \param epsilons                 [in/out] - radius to search, overwritten if uniform_radius != 0
            \param uniform_radius           [in] - double != 0 determines whether to overwrite all epsilons for uniform search
            \param max_search_radius        [in] - largest valid search (useful only for MPI jobs if halo size exists)
            \param check_same               [in] - ignored (to match nanoflann)
            \param do_dry_run               [in] - ignored (to match nanoflann)
        */
        template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
        size_t generateCRNeighborListsFromRadiusSearch(trg_view_type trg_pts_view, 
                neighbor_lists_view_type& neighbor_lists, neighbor_lists_view_type& number_of_neighbors_list, 
                epsilons_view_type& epsilons, const double uniform_radius = 0.0, 
                double max_search_radius = 0.0, bool check_same = true, bool do_dry_run = true) {

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
            if (number_of_neighbors_list.extent(0)!=(size_t)num_target_sites) {
                Kokkos::resize(number_of_neighbors_list, num_target_sites);
            }
            if (epsilons.extent(0)!=(size_t)num_target_sites) {
                Kokkos::resize(epsilons, num_target_sites);
            }
            compadre_assert_release((neighbor_lists_view_type::rank==1) && "neighbor_lists must be a 1D Kokkos view.");

            // perform tree search on device
            Kokkos::View<int*, device_memory_space> d_values("values", 0);
            Kokkos::View<int*, device_memory_space> d_offsets("offsets", 0);
            auto predicates = Radius<trg_view_type,epsilons_view_type>(
                    trg_pts_view, epsilons,
                    uniform_radius, max_search_radius);
            _tree.query(device_execution_space(), predicates, d_values, d_offsets);
            // get updated epsilons from device
            predicates.getRadiiFromDevice(epsilons);
            // move solution back to host accessible space 
            // (does nothing if already host accessible)
            auto values  = create_mirror_view(d_values);
            auto offsets = create_mirror_view(d_offsets);
            Kokkos::deep_copy(values,  d_values);
            Kokkos::deep_copy(offsets, d_offsets);
            Kokkos::fence();

            // set number of neighbors list (how many neighbors for each target site) based
            // on results on tree query
            Kokkos::parallel_for(Kokkos::RangePolicy<host_execution_space>(0, num_target_sites),
            KOKKOS_LAMBDA(int i) {
                size_t neighbors_found_for_target_i = offsets(i+1) - offsets(i);
                number_of_neighbors_list(i) = neighbors_found_for_target_i;
            });
            Kokkos::fence();

            // sort to put closest neighbor first (other neighbors are not affected)
            typedef Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
                    scratch_double_view;
            int team_scratch_size = 0;
            team_scratch_size += scratch_double_view::shmem_size(trg_pts_view.extent(1)); // distances
            Kokkos::parallel_for("first entry sort", host_team_policy(num_target_sites, Kokkos::AUTO)
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
                float first_neighbor_distance = kdtreeDistance(trg_pt.data(), values(offsets(i)));
                for (int j=1; j<number_of_neighbors_list(i); ++j) {
                    // distance to neighbor
                    float neighbor_distance = kdtreeDistance(trg_pt.data(), values(offsets(i)+j));
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
            if (neighbor_lists.extent(0) != values.extent(0)) {
                Kokkos::resize(neighbor_lists, values.extent(0));
            }
            Kokkos::deep_copy(neighbor_lists, values);
            Kokkos::fence();

            auto nla = CreateNeighborLists(number_of_neighbors_list);
            return nla.getTotalNeighborsOverAllListsHost();
        }


        /*! \brief Generates neighbor lists as 2D view by performing a k-nearest neighbor search
            Only accepts 2D neighbor_lists without number_of_neighbors_list.
            \param trg_pts_view             [in] - target coordinates from which to seek neighbors
            \param neighbor_lists           [out] - 2D view of neighbor lists to be populated from search
            \param epsilons                 [in/out] - radius to search, overwritten if uniform_radius != 0
            \param neighbors_needed         [in] - k neighbors needed as a minimum
            \param epsilon_multiplier       [in] - distance to kth neighbor multiplied by epsilon_multiplier for follow-on radius search
            \param max_search_radius        [in] - largest valid search (useful only for MPI jobs if halo size exists)
            \param check_same               [in] - ignored (to match nanoflann)
            \param do_dry_run               [in] - ignored (to match nanoflann)
        */
        template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
        size_t generate2DNeighborListsFromKNNSearch(trg_view_type trg_pts_view, 
                neighbor_lists_view_type& neighbor_lists, epsilons_view_type& epsilons, 
                const int neighbors_needed, const double epsilon_multiplier = 1.6, 
                double max_search_radius = 0.0, bool check_same = true, bool do_dry_run = true) {

            const int num_target_sites = trg_pts_view.extent(0);
 
            // check neighbor lists rank
            compadre_assert_release((neighbor_lists_view_type::rank==2) && "neighbor_lists must be a 2D Kokkos view.");

            // temporary 1D array for cr_neighbor_lists
            typedef typename neighbor_lists_view_type::value_type value_type;
            Kokkos::View<value_type*, typename neighbor_lists_view_type::memory_space> 
                cr_neighbor_lists(neighbor_lists.label(), 0);

            // temporary 1D array for number_of_neighbors_list
            typedef typename neighbor_lists_view_type::value_type value_type;
            Kokkos::View<value_type*, typename neighbor_lists_view_type::memory_space> 
                number_of_neighbors_list("number of neighbors list", num_target_sites);

            generateCRNeighborListsFromKNNSearch(trg_pts_view,
                cr_neighbor_lists, number_of_neighbors_list, epsilons, 
                neighbors_needed, epsilon_multiplier, max_search_radius,
                check_same, do_dry_run);

            auto nla = CreateNeighborLists(cr_neighbor_lists, number_of_neighbors_list);
            size_t max_num_neighbors = nla.getMaxNumNeighbors();
            ConvertCompressedRowNeighborListsTo2D(nla, neighbor_lists);

            return max_num_neighbors;
        }

        /*! \brief Generates compressed row neighbor lists by performing a k-nearest neighbor search
            Only accepts 1D neighbor_lists with 1D number_of_neighbors_list.
            \param trg_pts_view             [in] - target coordinates from which to seek neighbors
            \param neighbor_lists           [out] - 1D view of neighbor lists to be populated from search
            \param number_of_neighbors_list [in/out] - number of neighbors for each target site
            \param epsilons                 [in/out] - radius to search, overwritten if uniform_radius != 0
            \param neighbors_needed         [in] - k neighbors needed as a minimum
            \param epsilon_multiplier       [in] - distance to kth neighbor multiplied by epsilon_multiplier for follow-on radius search
            \param max_search_radius        [in] - largest valid search (useful only for MPI jobs if halo size exists)
            \param check_same               [in] - ignored (to match nanoflann)
            \param do_dry_run               [in] - ignored (to match nanoflann)
        */
        template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
        size_t generateCRNeighborListsFromKNNSearch(trg_view_type trg_pts_view, 
                neighbor_lists_view_type& neighbor_lists, neighbor_lists_view_type& number_of_neighbors_list,
                epsilons_view_type& epsilons, const int neighbors_needed, const double epsilon_multiplier = 1.6, 
                double max_search_radius = 0.0, bool check_same = true, bool do_dry_run = true) {

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

            if (number_of_neighbors_list.extent(0)!=(size_t)num_target_sites) {
                Kokkos::resize(number_of_neighbors_list, num_target_sites);
            }
            if (epsilons.extent(0)!=(size_t)num_target_sites) {
                Kokkos::resize(epsilons, num_target_sites);
            }
            compadre_assert_release((neighbor_lists_view_type::rank==1) && "neighbor_lists must be a 1D Kokkos view.");

            // perform tree search on device
            Kokkos::View<int*, device_memory_space> d_values("values", 0);
            Kokkos::View<int*, device_memory_space> d_offsets("offsets", 0);
            auto predicates = NearestNeighbor<trg_view_type>(trg_pts_view, neighbors_needed);
            _tree.query(device_execution_space(), predicates, d_values, d_offsets);
            // move solution back to host accessible space 
            // (does nothing if already host accessible)
            auto values  = create_mirror_view(d_values);
            auto offsets = create_mirror_view(d_offsets);
            Kokkos::deep_copy(values,  d_values);
            Kokkos::deep_copy(offsets, d_offsets);
            Kokkos::fence();

            size_t min_num_neighbors = 0;
            // if no target sites, then min_num_neighbors is set to neighbors_needed
            if (num_target_sites==0) min_num_neighbors = neighbors_needed;

            // get min_num_neighbors
            typedef Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
                    scratch_double_view;
            int team_scratch_size = 0;
            team_scratch_size += scratch_double_view::shmem_size(trg_pts_view.extent(1)); // distances
            Kokkos::parallel_reduce("knn inspect", host_team_policy(num_target_sites, Kokkos::AUTO)
                    .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)), 
                    KOKKOS_LAMBDA(const host_member_type& teamMember, size_t& t_min_num_neighbors) {

                const int i = teamMember.league_rank();

                size_t num_neighbors = offsets(i+1) - offsets(i);
                t_min_num_neighbors = (num_neighbors < t_min_num_neighbors) ?
                                        num_neighbors : t_min_num_neighbors;

                // make unmanaged scratch views
                scratch_double_view trg_pt(teamMember.team_scratch(0 /*shared memory*/), trg_pts_view.extent(1));
                for (int j=0; j<trg_pts_view.extent(1); ++j) {
                    trg_pt(j) = trg_pts_view(i,j);
                }
                search_scalar last_neighbor_distance = kdtreeDistance(trg_pt.data(), values(offsets(i)+num_neighbors-1));

                // scale by epsilon_multiplier to window from location where the last neighbor was found
                epsilons(i) = static_cast<search_scalar>(last_neighbor_distance*epsilon_multiplier);
                // because radius search is inclusive of endpoints for ArborX, no need to scale zero distances
                // as an epsilon value times the epsilon multiplier (like is done in Nanoflann where radius 
                // searches do not pick up points radius distance from target)

                compadre_kernel_assert_release((epsilons(i)<=max_search_radius || max_search_radius==0) 
                        && "max_search_radius given (generally derived from the size of a halo region), \
                            and search radius needed would exceed this max_search_radius.");

#ifdef COMPADRE_DEBUG
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_neighbors-1), [=](const int j) {
                    search_scalar this_distance = kdtreeDistance(trg_pt.data(), values(offsets(i)+j  ));
                    search_scalar next_distance = kdtreeDistance(trg_pt.data(), values(offsets(i)+j+1));
                    compadre_kernel_assert_debug((this_distance-next_distance<SEARCH_SCALAR_EPS) 
                        && "Neighbors returned by KD tree search are not ordered.");
                });
#endif
            }, Kokkos::Min<size_t>(min_num_neighbors));

            // Next, check that we found the neighbors_needed number that we require for unisolvency
            compadre_assert_release((num_target_sites==0 || (min_num_neighbors>=(size_t)neighbors_needed))
                    && "Neighbor search failed to find number of neighbors needed for unisolvency.");
            
            // call a radius search using values now stored in epsilons
            generateCRNeighborListsFromRadiusSearch(trg_pts_view, neighbor_lists, 
                    number_of_neighbors_list, epsilons, 0.0 /*don't set uniform radius*/, 
                    max_search_radius, check_same, do_dry_run);

            auto nla = CreateNeighborLists(number_of_neighbors_list);
            return nla.getTotalNeighborsOverAllListsHost();
        }
}; // ArborX's PointCloudSearch

} // Compadre

#endif
