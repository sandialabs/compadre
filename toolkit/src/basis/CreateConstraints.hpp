#ifndef _CREATE_CONSTRAINTS_
#define _CREATE_CONSTRAINTS_

#include "Compadre_GMLS.hpp"

namespace Compadre {

KOKKOS_INLINE_FUNCTION
void evaluateConstraints(scratch_matrix_right_type M, scratch_matrix_right_type PsqrtW, const ConstraintType constraint_type, const double cutoff_p, const int num_neighbors = 0, scratch_matrix_right_type* T = NULL) {
    if (constraint_type == ConstraintType::NEUMANN_GRAD_SCALAR) {
        // Fill in the bottom right entry for PsqrtW
        PsqrtW(num_neighbors, PsqrtW.extent(1)-1) = 1.0;

        // Fill in the last column and row of M
        M(M.extent(0)-1, 1) = (1.0/cutoff_p)*(*T)(2,0);
        M(1, M.extent(0)-1) = (1.0/cutoff_p)*(*T)(2,0);

        M(M.extent(0)-1, 2) = (1.0/cutoff_p)*(*T)(2,1);
        M(2, M.extent(0)-1) = (1.0/cutoff_p)*(*T)(2,1);

        M(M.extent(0)-1, 3) = (1.0/cutoff_p)*(*T)(2,2);
        M(3, M.extent(0)-1) = (1.0/cutoff_p)*(*T)(2,2);
    }
}

}
#endif
