#ifndef _COMPADRE_POINTCLOUDSEARCH_HPP_
#define _COMPADRE_POINTCLOUDSEARCH_HPP_

#include "Compadre_Typedefs.hpp"
#include "Compadre_NeighborLists.hpp"

#ifdef COMPADRE_USE_ARBORX
  #include "Compadre_PointCloudSearch_ArborX.hpp"
#else
  #include "Compadre_PointCloudSearch_Nanoflann.hpp"
#endif
#include <Kokkos_Core.hpp>

namespace Compadre {

template <typename view_type>
template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
void PointCloudSearch<view_type>::generate2DNeighborListsFromRadiusSearch(trg_view_type trg_pts_view,
        neighbor_lists_view_type& neighbor_lists, epsilons_view_type& epsilons,
        const double uniform_radius, double max_search_radius,
        bool check_same) {

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
        max_search_radius, check_same, true /*do_dry_run*/);

    auto nla = CreateNeighborLists(cr_neighbor_lists, number_of_neighbors_list);
    size_t max_num_neighbors = nla.getMaxNumNeighbors();
    ConvertCompressedRowNeighborListsTo2D(nla, neighbor_lists);
}

template <typename view_type>
template <typename trg_view_type, typename neighbor_lists_view_type, typename epsilons_view_type>
void PointCloudSearch<view_type>::generate2DNeighborListsFromKNNSearch(trg_view_type trg_pts_view,
        neighbor_lists_view_type& neighbor_lists, epsilons_view_type& epsilons,
        const int neighbors_needed, const double epsilon_multiplier,
        double max_search_radius, bool check_same) {

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
        check_same);

    auto nla = CreateNeighborLists(cr_neighbor_lists, number_of_neighbors_list);
    size_t max_num_neighbors = nla.getMaxNumNeighbors();
    ConvertCompressedRowNeighborListsTo2D(nla, neighbor_lists);
}

//! CreatePointCloudSearch allows for the construction of an object of type PointCloudSearch with template deduction
template <typename view_type>
PointCloudSearch<view_type> CreatePointCloudSearch(view_type src_view, const local_index_type dimensions = -1, const local_index_type max_leaf = -1) { 
    return PointCloudSearch<view_type>(src_view, dimensions, max_leaf);
}

} // Compadre

#endif
