#ifndef _CREATE_CONSTRAINTS_
#define _CREATE_CONSTRAINTS_

#include "Compadre_GMLS.hpp"

namespace Compadre {

KOKKOS_INLINE_FUNCTION
void evaluateConstraints(scratch_matrix_right_type M, scratch_matrix_right_type PsqrtW, const ConstraintType constraint_type, const double cutoff_p, const int dimension, const int num_neighbors = 0, scratch_matrix_right_type* T = NULL) {
    if (constraint_type == ConstraintType::NEUMANN_GRAD_SCALAR) {
        // Fill in the bottom right entry for PsqrtW
        PsqrtW(num_neighbors, PsqrtW.extent(1)-1) = 1.0;

        // Fill in the last column and row of M
        for (int i=0; i<dimension; ++i) {
            M(M.extent(0)-1, i+1) = (1.0/cutoff_p)*(*T)(dimension-1,i);
            M(i+1, M.extent(0)-1) = (1.0/cutoff_p)*(*T)(dimension-1,i);
        }
    }
}

}
#endif
