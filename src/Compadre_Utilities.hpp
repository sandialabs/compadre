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

}  // Compadre namespace

#endif
