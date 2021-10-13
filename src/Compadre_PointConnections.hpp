#ifndef _COMPADRE_POINTCONNECTIONS_HPP_
#define _COMPADRE_POINTCONNECTIONS_HPP_

#include "Compadre_Typedefs.hpp"
#include <Kokkos_Core.hpp>

namespace Compadre {

//!  Combines NeighborLists with the PointClouds from which it was derived
template <typename view_type_1, typename view_type_2, typename view_type_3>
class PointConnections {
public:

    //! source site coordinates on device
    typedef decltype(Kokkos::create_mirror_view<device_memory_space>(
                device_memory_space(), view_type_1()))
                        device_mirror_target_view_type;

    //! target site coordinates on device
    typedef decltype(Kokkos::create_mirror_view<device_memory_space>(
                device_memory_space(), view_type_2()))
                        device_mirror_source_view_type;

    device_mirror_target_view_type _target_coordinates;
    device_mirror_source_view_type _source_coordinates;
    NeighborLists<view_type_3> _nla;

/** @name Constructors
 */
///@{

    //! \brief Constructor for PointConnections
    PointConnections(Kokkos::View<view_type_1> target_coordinates, 
                     Kokkos::View<view_type_2> source_coordinates,
                     NeighborLists<view_type_3> nla) : {

        _target_coordinates = Kokkos::create_mirror_view<device_memory_space>(
                device_memory_space(), view_type_1())
        _source_coordinates = Kokkos::create_mirror_view<device_memory_space>(
                device_memory_space(), view_type_2())
        Kokkos::deep_copy(_target_coordinates, target_coordinates);
        Kokkos::deep_copy(_source_coordinates, source_coordinates);

    }

///@}

}; // PointConnections

} // Compadre namespace

#endif

