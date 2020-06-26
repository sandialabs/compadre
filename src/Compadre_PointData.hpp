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

    // needed for copy constructor between template types
    template<class view_type_2>
    friend class PointData;

    bool _needs_sync_to_host;
    int _number_of_coordinates;
    int _dimension;

    view_type _coordinates;

    typename view_type::HostMirror _host_coordinates;

    void initialize(view_type coordinates) {

        _number_of_coordinates = coordinates.extent(0);
        _dimension = coordinates.extent(1);

        _coordinates = coordinates;

        _host_coordinates = Kokkos::create_mirror_view(_coordinates);
        Kokkos::deep_copy(_host_coordinates, _coordinates);
        Kokkos::fence();

        _needs_sync_to_host = false;

    }
   

public:

/** @name Constructors
 *  Ways to initialize a PointData object
 */
///@{

    //! \brief Default constructor for the purpose of classes who have PointData as a member object
    PointData() {

        _needs_sync_to_host = true;
        _number_of_coordinates = 0;
        _dimension = 0;

    }

    //! \brief Constructor for coordinates type matching internal view type
    template<typename view_type_2, typename std::enable_if<view_type_2::rank==2&&std::is_same<view_type,view_type_2>::value==1, int>::type = 0 > 
    PointData(view_type_2 coordinates) {

        printf("same view type.\n");
        this->initialize(coordinates);

    }

    //! \brief Constructor for coordinates type NOT matching internal view type (different memory space)
    template<typename view_type_2, typename std::enable_if<view_type_2::rank==2
                                                           &&std::is_same<view_type,view_type_2>::value==0
                                                           &&Kokkos::SpaceAccessibility<typename view_type::execution_space, typename view_type_2::memory_space>::accessible==0, 
                                                                int>::type = 0 >
    PointData(view_type_2 coordinates) {

        printf("called for diff space.\n");
        decltype(_coordinates) tmp_coordinates(coordinates.label(), coordinates.extent(0), coordinates.extent(1));
        auto view_2_space_mirror = Kokkos::create_mirror_view(typename view_type_2::memory_space(), tmp_coordinates);
        Kokkos::deep_copy(view_2_space_mirror, coordinates);
        Kokkos::fence();
        Kokkos::deep_copy(tmp_coordinates, view_2_space_mirror);
        Kokkos::fence();

        this->initialize(tmp_coordinates);

    }

    //! \brief Constructor for coordinates type NOT matching internal view type (same/accessible memory space)
    template<typename view_type_2, typename std::enable_if<view_type_2::rank==2
                                                           &&std::is_same<view_type,view_type_2>::value==0
                                                           &&Kokkos::SpaceAccessibility<typename view_type::execution_space, typename view_type_2::memory_space>::accessible==1, 
                                                                int>::type = 0 >
    PointData(view_type_2 coordinates) {

        printf("called for same space.\n");
        decltype(_coordinates) tmp_coordinates(coordinates.label(), coordinates.extent(0), coordinates.extent(1));
        Kokkos::deep_copy(tmp_coordinates, coordinates);
        Kokkos::fence();

        this->initialize(tmp_coordinates);

    }

    //! \brief Default copy constructor for same template, same class (shallow copy)
    PointData(const PointData&) = default;

    //! \brief Copy constructor for another template instantiation of same class
    //  Intentionally const because the end result will be deep-copied coordinate data.
    template<typename view_type_2, typename std::enable_if<std::is_same<view_type,view_type_2>::value==0, int>::type = 0>
    PointData(const PointData<view_type_2>& other) {
    
        printf("two instance copy constructor called.\n");
        PointData<view_type> pd(other._coordinates);
        // default assignment operator (same template type)
        *this = pd;
    
    }

    //! \brief Default assignment operator for same template, same class (shallow copy)
    PointData& operator=(const PointData&) = default;

    //! \brief Assignment operator for another template instantiation of same class
    //  Intentionally const because the end result will be deep-copied coordinate data.
    template<typename view_type_2, typename std::enable_if<std::is_same<view_type,view_type_2>::value==0, int>::type = 0>
    PointData& operator=(const PointData<view_type_2>& other) {
    
        printf("two instance assignment operator called.\n");
        PointData<view_type> pd(other._coordinates);
        // default assignment operator (same template type)
        *this = pd;
        return *this;
    
    }

///@}

/** @name Public accessors
 */
///@{
    double getCoordinateHost(const int index, const int dim, const scratch_matrix_right_type* V = NULL) const {
        compadre_kernel_assert_debug((_number_of_coordinates >= index) && "Index is out of range for coordinates.");
        if (V==NULL) {
            return _host_coordinates(index, dim);
        } else {
            XYZ coord;
            switch ((*V).extent(0)) {
                case 3:
                    coord = XYZ( _host_coordinates(index, 0),
                                 _host_coordinates(index, 1),
                                 _host_coordinates(index, 2) );
                    return convertGlobalToLocalCoordinate(coord, dim, V);
                case 2:
                    coord = XYZ( _host_coordinates(index, 0),
                                 _host_coordinates(index, 1) );
                    return convertGlobalToLocalCoordinate(coord, dim, V);
                default:
                    coord = XYZ( _host_coordinates(index, 0) );
                    return convertGlobalToLocalCoordinate(coord, dim, V);

            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    double getCoordinateDevice(const int index, const int dim, const scratch_matrix_right_type* V = NULL) const {
        compadre_kernel_assert_debug((_number_of_coordinates >= index) && "Index is out of range for coordinates.");
        if (V==NULL) {
            return _coordinates(index, dim);
        } else {
            XYZ coord;
            switch ((*V).extent(0)) {
                case 3:
                    coord = XYZ( _coordinates(index, 0),
                                 _coordinates(index, 1),
                                 _coordinates(index, 2) );
                    return convertGlobalToLocalCoordinate(coord, dim, V);
                case 2:
                    coord = XYZ( _coordinates(index, 0),
                                 _coordinates(index, 1) );
                    return convertGlobalToLocalCoordinate(coord, dim, V);
                default:
                    coord = XYZ( _coordinates(index, 0) );
                    return convertGlobalToLocalCoordinate(coord, dim, V);

            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    size_t getNumberOfCoordinates() const {
        return _number_of_coordinates;
    }

///@}

}; // PointData

} // Compadre

#endif
