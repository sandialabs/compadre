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

    bool _host_fill;
    bool _device_fill;

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

        _host_fill = false;
        _device_fill = false;

    }

   

public:

/** @name Constructors
 *  Ways to initialize a PointData object
 */
///@{

    //! \brief Default constructor for the purpose of classes who have PointData as a member object
    PointData() {

        _host_fill = false;
        _device_fill = false;
        _number_of_coordinates = 0;
        _dimension = 0;

    }

    //! \bried Constructor for certain size
    PointData(const std::string& label, size_t dimension_1, size_t dimension_2) {

        decltype(_coordinates) tmp_coordinates(label, dimension_1, dimension_2);
        Kokkos::deep_copy(tmp_coordinates, 0.0);
        Kokkos::fence();
        this->initialize(tmp_coordinates);

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

/** @name Public modifiers
 */
///@{
    
    // philosophy is that setting values on device should be fast, because these are called from a device kernel and the user
    // who will call this is likely interested in speed
    // setting and accessing values on host can be slower (check that info is synced with device), because the fact that it is
    // being called from host default execution space already indicates less interest in performance
    //
    // for the reasons above, resumeFill must be called to fill values on this host (and will be checked for all
    // set functions on host).
    // 
    // when running in debug mode, fill modes are checked for both host and device
  
    void resumeFillOnHost() {
        compadre_assert_release(!_device_fill && "resumeFillOnHost() called while resumeFillOnDevice() still active\n");
        _host_fill = true;
    }

    void resumeFillOnDevice() {
        compadre_assert_release(!_host_fill && "resumeFillOnDevice() called while resumeFillOnHost() still active\n");
        _device_fill = true;
    }

    void fillCompleteOnHost() {
        compadre_assert_release(_host_fill && "fillCompleteOnHost() called while before / multiple times after call to resumeFillOnHost()\n");
        Kokkos::deep_copy(_coordinates, _host_coordinates);
        Kokkos::fence();
        _host_fill = false;
    }

    void fillCompleteOnDevice() {
        compadre_assert_release(_device_fill && "fillCompleteOnDevice() called while before / multiple times after call to resumeFillOnDevice()\n");
        Kokkos::deep_copy(_host_coordinates, _coordinates);
        Kokkos::fence();
        _device_fill = false;
    }

    typename std::enable_if<view_type::rank==2, void>::type 
            setValueOnHost(int i, int j, typename view_type::value_type k) {

        compadre_assert_release(_host_fill && "setValueOnHost called without resumeFillOnHost() being active\n");
        _host_coordinates(i,j) = k;

    }

    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<view_type::rank==2, void>::type 
            setValueOnDevice(int i, int j, typename view_type::value_type k) {

        compadre_kernel_assert_debug(_device_fill && "setValueOnDevice called without resumeFillOnDevice() being active\n");
        _coordinates(i,j) = k;

    }

    void replaceWith(std::vector<std::vector<typename view_type::data_type> > std_vec_values, const std::string& label = std::string()) {

        compadre_assert_release((view_type::rank==2) && "Invalid input to replaceWith for relative to rank of PointData\n");

        decltype(_coordinates) tmp_coordinates(label, std_vec_values.size(), std_vec_values[0].size());
        auto tmp_coordinates_mirror = create_mirror_view(tmp_coordinates);
        for (size_t i=0; i<std_vec_values.size(); ++i) {
            for (size_t j=0; j<std_vec_values[0].size(); ++j) {
                tmp_coordinates_mirror(i,j) = std_vec_values[i][j];
            }
        }
        Kokkos::deep_copy(tmp_coordinates, tmp_coordinates_mirror);
        Kokkos::fence();
        this->initialize(tmp_coordinates);

    }

    void replaceWith(std::vector<std::vector<std::vector<typename view_type::data_type> > > std_vec_values, const std::string& label = std::string()) {

        compadre_assert_release((view_type::rank==3) && "Invalid input to replaceWith for relative to rank of PointData\n");

        decltype(_coordinates) tmp_coordinates(label, std_vec_values.size(), std_vec_values[0].size(), std_vec_values[0][0].size());
        auto tmp_coordinates_mirror = create_mirror_view(tmp_coordinates);
        for (size_t i=0; i<std_vec_values.size(); ++i) {
            for (size_t j=0; j<std_vec_values[0].size(); ++j) {
                for (size_t k=0; k<std_vec_values[0][0].size(); ++k) {
                    tmp_coordinates_mirror(i,j,k) = std_vec_values[i][j][k];
                }
            }
        }
        Kokkos::deep_copy(tmp_coordinates, tmp_coordinates_mirror);
        Kokkos::fence();
        this->initialize(tmp_coordinates);

    }

    void replaceWith(view_type coordinates) {

        this->initialize(coordinates);

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
    size_t getNumberOfPoints() const {
        return _number_of_coordinates;
    }

    typename std::enable_if<view_type::rank==2, typename view_type::value_type>::type getValueOnHost(int i, int j) const {
        compadre_assert_release((!_host_fill && !_device_fill) && "getValueOnHost called while resumeFillOnHost() or resumeFillOnDevice() being active\n");
        return _host_coordinates(i,j);
    }

    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<view_type::rank==2, typename view_type::value_type>::type getValueOnDevice(int i, int j) const {
        compadre_kernel_assert_debug((!_host_fill && !_device_fill) && "getValueOnHost called while resumeFillOnHost() or resumeFillOnDevice() being active\n");
        return _coordinates(i,j);
    }

    KOKKOS_INLINE_FUNCTION
    int getDimension() const {
        return _dimension;
    }

    size_t extent(int i) const {
        return _coordinates.extent(i);
    }

    bool isFillActiveOnHost() const {
        return _host_fill;
    }

    KOKKOS_INLINE_FUNCTION
    bool isFillActiveOnDevice() const {
        return _device_fill;
    }

    std::string getLabel() const {
        return _coordinates.label();
    }


///@}

}; // PointData

template <typename view_type>
std::ostream& operator << ( std::ostream& os, const PointData<view_type>& pd ) {
    os << "Details for PointData(" << pd.getLabel() << "):" << std::endl;
    os << "======================================================" << std::endl;
    os << "_host_fill: " << pd.isFillActiveOnHost() << std::endl;
    os << "_device_fill: " << pd.isFillActiveOnDevice() << std::endl;
    os << "address in memory: " << &pd << std::endl;
    os << "length: " << pd.getNumberOfPoints() << std::endl;
    return os; 
}

} // Compadre

#endif
