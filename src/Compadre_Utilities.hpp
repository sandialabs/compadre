#ifndef _COMPADRE_UTILITIES_HPP_
#define _COMPADRE_UTILITIES_HPP_

#include "Compadre_Config.h"
#include "Compadre_Typedefs.hpp"

namespace Compadre {

KOKKOS_INLINE_FUNCTION
void getMidpointFromCellVertices(const member_type& teamMember, scratch_vector_type midpoint_storage, scratch_matrix_right_type cell_coordinates, const int cell_num, const int dim=3) {
Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
    auto num_nodes = cell_coordinates.extent(1)/dim;
    for (int j=0; j<dim; ++j) {
        midpoint_storage(j) = 0;
        for (int i=0; i<num_nodes; ++i) {
            midpoint_storage(j) += cell_coordinates(cell_num, i*dim + j) / (double)(num_nodes);
        }
    }
});
}

template <typename output_memory_space, typename view_type_input_data, typename output_array_layout = typename view_type_input_data::array_layout, typedef index_type = int>
Kokkos::View<int*, output_array_layout, output_memory_space> // shares layout of input by default
        filterViewByID(view_type_input_data input_data_host_or_device, index_type filtered_value) {

    // Make view on the device (does nothing if already on the device)
    auto input_data_device = Kokkos::create_mirror_view(
        device_execution_space::memory_space(), input_data_host_or_device);
    Kokkos::deep_copy(input_data_device, input_data_host_or_device);
    Kokkos::fence();

    // Count the number of elements in the input view that match the desired value
    int num_count = 0;
    Kokkos::parallel_reduce("Count occurrences of values in the view", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace(0, input_data_device.extent(0)), KOKKOS_LAMBDA(const int i, int& local_num_count) {
        if (input_data_device(i) == desired_value) {
            local_num_count++;
        }
    }, num_count);
    Kokkos::fence();

    // Create a new view living on device
    Kokkos::View<int*, output_array_layout> filtered_view("filterd view", num_count);
    // Gather up the indices into the new view
    int filtered_index = 0;
    for (int i=0; i<input_data_device(i).extent(0); i++) {
        if (input_data_device(i) == filtered_value) {
            filtered_view(filtered_index) = i;
            filtered_index++;
        }
    }
    Kokkos::fence();

    // Then copy it back out - either to host or device space based on user's request
    typedef Kokkos::View<int*, output_array_layout, output_memory_space> output_view_type;
    output_view_type filtered_view_output("output filtered view", num_count);
    Kokkos::deep_copy(filtered_view_output, filtered_view);
    Kokkos::fence();

    return filtered_view_output;
}

}  // Compadre namespace

#endif
