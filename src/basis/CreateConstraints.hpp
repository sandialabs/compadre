#ifndef _CREATE_CONSTRAINTS_
#define _CREATE_CONSTRAINTS_

#include "Compadre_GMLS.hpp"

namespace Compadre {

KOKKOS_INLINE_FUNCTION
void evaluateConstraints(scratch_matrix_right_type M, scratch_matrix_right_type PsqrtW, const ConstraintType constraint_type, const ReconstructionSpace reconstruction_space, const int NP, const double cutoff_p, const int dimension, const int num_neighbors = 0, scratch_matrix_right_type* T = NULL) {
    if (constraint_type == ConstraintType::NEUMANN_GRAD_SCALAR) {
        if (reconstruction_space == ReconstructionSpace::ScalarTaylorPolynomial) {
            // Fill in the bottom right entry for PsqrtW
            PsqrtW(num_neighbors, PsqrtW.extent(1)-1) = 1.0;

            // Fill in the last column and row of M
            for (int i=0; i<dimension; ++i) {
                M(M.extent(0)-1, i+1) = (1.0/cutoff_p)*(*T)(dimension-1,i);
                M(i+1, M.extent(0)-1) = (1.0/cutoff_p)*(*T)(dimension-1,i);
            }
        } else if (reconstruction_space == ReconstructionSpace::VectorTaylorPolynomial) {
            // Fill in the bottom right of PsqrtW
            PsqrtW(num_neighbors, PsqrtW.extent(1) - 1) = 1.0;
            PsqrtW(num_neighbors, PsqrtW.extent(1) - 2) = 1.0;
            PsqrtW(num_neighbors, PsqrtW.extent(1) - 3) = 1.0;

            // Fill in the last column and row of M
            M(0, M.extent(0)-3) = (*T)(2, 0);
            M(0, M.extent(0)-2) = (*T)(2, 0);
            M(0, M.extent(0)-1) = (*T)(2, 0);
            M(M.extent(0)-3, 0) = (*T)(2, 0);
            M(M.extent(0)-2, 0) = (*T)(2, 0);
            M(M.extent(0)-1, 0) = (*T)(2, 0);

            M(NP, M.extent(0) - 3) = (*T)(2, 1);
            M(NP, M.extent(0) - 2) = (*T)(2, 1);
            M(NP, M.extent(0) - 1) = (*T)(2, 1);
            M(M.extent(0) - 3, NP) = (*T)(2, 1);
            M(M.extent(0) - 2, NP) = (*T)(2, 1);
            M(M.extent(0) - 1, NP) = (*T)(2, 1);

            M(2*NP, M.extent(0) - 3) = (*T)(2, 2);
            M(2*NP, M.extent(0) - 2) = (*T)(2, 2);
            M(2*NP, M.extent(0) - 1) = (*T)(2, 2);
            M(M.extent(0) - 3, 2*NP) = (*T)(2, 2);
            M(M.extent(0) - 2, 2*NP) = (*T)(2, 2);
            M(M.extent(0) - 1, 2*NP) = (*T)(2, 2);
        }
    }
}

}
#endif
