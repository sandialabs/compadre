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

template <typename view_type_1, typename view_type_2>
KOKKOS_INLINE_FUNCTION
double getAreaFromVectors(const member_type& teamMember, view_type_1 v1, view_type_2 v2) {
    double area = 0;
    double val = v1[1]*v2[2] - v1[2]*v2[1];
    area += val*val;
    val = v1[2]*v2[0] - v1[0]*v2[2];
    area += val*val;
    val = v1[0]*v2[1] - v1[1]*v2[0];
    area += val*val;
    return std::sqrt(area);
}

}  // Compadre namespace

#endif
