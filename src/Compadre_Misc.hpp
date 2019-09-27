#ifndef _COMPADRE_MISC_HPP_
#define _COMPADRE_MISC_HPP_

#include "Compadre_Operators.hpp"

namespace Compadre {

struct XYZ {

    KOKKOS_INLINE_FUNCTION
    XYZ() : x(0), y(0), z(0) {}

    KOKKOS_INLINE_FUNCTION
    XYZ(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    double x;
    double y;
    double z;

    KOKKOS_INLINE_FUNCTION
    double& operator [](const int i) {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            default:
                return z;
        }
    }

    KOKKOS_INLINE_FUNCTION
    XYZ operator *(double scalar) {
        XYZ result;
        result.x = scalar*x;
        result.y = scalar*y;
        result.z = scalar*z;
        return result;
    }
}; // XYZ

KOKKOS_INLINE_FUNCTION
// Get the first argument to be teamMember (check other prewritten functors)
// change the order of argument: first memory on scratch, then memory on global
void convertMatrixLayoutRightToLeft(scratch_matrix_right_type global_mat, scratch_matrix_left_type scratch_mat) {
    // create a layout left view on the global matrix
    scratch_matrix_left_type global_mat_left(global_mat.data(), global_mat.extent(0), global_mat.extent(1));

    // Create a 1D array view on the global matrix data
    scratch_vector_type global_mat_left_flat(global_mat_left.data(), global_mat_left.extent(0)*global_mat_left.extent(1));
    // Create a 1D array view on the scratch matrix data
    scratch_vector_type scratch_mat_flat(scratch_mat.data(), scratch_mat.extent(0)*scratch_mat.extent(1));

    // Use a kokkos single first to check working in serial
    // then use kokkos parallel_for later
    // Transpose the matrix and copy from the global mat data to scratch mat data
    for (int i=0; i<scratch_mat.extent(0); i++) {
        for (int j=0; j<scratch_mat.extent(1); j++) {
            // Transpose the global matrix and copy into scratch matrix data
            scratch_mat(i, j) = global_mat_left(j, i);
        }
    }

    // Now copy the flat 1D array from scratch matrix data back to the global data
    for (int i=0; i<scratch_mat.extent(0)*scratch_mat.extent(1); i++) {
        global_mat_left_flat(i) = scratch_mat_flat(i);
    }
}

}; // Compadre

#endif
