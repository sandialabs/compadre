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

//! CreatePointCloudSearch allows for the construction of an object of type PointCloudSearch with template deduction
template <typename view_type>
PointCloudSearch<view_type> CreatePointCloudSearch(view_type src_view, const local_index_type dimensions = -1, const local_index_type max_leaf = -1) { 
    return PointCloudSearch<view_type>(src_view, dimensions, max_leaf);
}

} // Compadre

#endif
