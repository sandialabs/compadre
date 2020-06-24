#ifndef _COMPADRE_POINTDATA_HPP_
#define _COMPADRE_POINTDATA_HPP_

#include "Compadre_Typedefs.hpp"
#include <Kokkos_Core.hpp>

namespace Compadre {

//!  PointData assists in setting/accessing point data
template <typename view_type>
class PointData {
public:

    typedef view_type internal_view_type;

protected:

    bool _needs_sync_to_host;
    int _number_of_coordinates;
    int _dimension;

    view_type _coordinates;

    typename view_type::HostMirror _host_coordinates;

   

public:

/** @name Constructors
 *  Ways to initialize a PointData object
 */
///@{

    //! \brief Constructor for the purpose of classes who have PointData as a member object
    PointData() {
        _needs_sync_to_host = true;
        _number_of_coordinates = 0;
        _dimension = 3;
    }

    //template<typename view_type, std::enable_if<view_type_2::rank==2&&std::is_same<view_type,view_type_2>::value==1>::type >
    //template<typename view_type_2, std::enable_if<typename view_type_2::rank==2&&std::is_same<view_type,view_type_2>::value==1>::type > //, std::enable_if<view_type_2::rank==2&&std::is_same<view_type,view_type_2>::value==1>::type >
    template<typename view_type_2, typename std::enable_if<view_type_2::rank==2&&std::is_same<view_type,view_type_2>::value==1, int>::type = 0 > //, std::enable_if<view_type_2::rank==2&&std::is_same<view_type,view_type_2>::value==1>::type >
    PointData(view_type_2 coordinates) {

        printf("constructor 1\n");

        compadre_assert_release((view_type_2::rank==2) && 
                "coordinates must be a 2D Kokkos view.");

        _number_of_coordinates = coordinates.extent(0);
        _dimension = coordinates.extent(1);

        _coordinates = coordinates;

        _host_coordinates = coordinates;

        Kokkos::deep_copy(_host_coordinates, _coordinates);
        Kokkos::fence();

        _needs_sync_to_host = false;
    }

    template<typename view_type_2, typename std::enable_if<view_type_2::rank==2&&std::is_same<view_type,view_type_2>::value==0, int>::type = 0 >
    PointData(view_type_2 coordinates) {
        printf("constructor 2\n");

    }

};

} // Compadre

#endif
