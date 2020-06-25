#ifndef _COMPADRE_MISC_HPP_
#define _COMPADRE_MISC_HPP_

#include "Compadre_Operators.hpp"

namespace Compadre {

struct XYZ {

    KOKKOS_INLINE_FUNCTION
    XYZ() : x(0), y(0), z(0) {}

    KOKKOS_INLINE_FUNCTION
    XYZ(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    KOKKOS_INLINE_FUNCTION
    XYZ(double _x, double _y) : x(_x), y(_y), z(0) {}

    KOKKOS_INLINE_FUNCTION
    XYZ(double _x) : x(_x), y(0), z(0) {}

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

//! Returns a component of the local coordinate after transformation from global to local under the orthonormal basis V.
KOKKOS_INLINE_FUNCTION
double convertGlobalToLocalCoordinate(const XYZ global_coord, const int dim, const scratch_matrix_right_type* V) {
    // only written for up n-1 dim manifold in [n<=3] space
    double val = 0;
    val += global_coord.x * (*V)(dim, 0);
    if ((*V).extent(0)>1) val += global_coord.y * (*V)(dim, 1);
    if ((*V).extent(0)>2) val += global_coord.z * (*V)(dim, 2);
    return val;
}


KOKKOS_INLINE_FUNCTION
int getAdditionalAlphaSizeFromConstraint(DenseSolverType dense_solver_type, ConstraintType constraint_type) {
    // Return the additional constraint size
    if (constraint_type == NEUMANN_GRAD_SCALAR) {
        return 1;
    } else {
        return 0;
    }
}

KOKKOS_INLINE_FUNCTION
int getAdditionalCoeffSizeFromConstraintAndSpace(DenseSolverType dense_solver_type, ConstraintType constraint_type, ReconstructionSpace reconstruction_space, const int dimension) {
    // Return the additional constraint size
    int added_alpha_size = getAdditionalAlphaSizeFromConstraint(dense_solver_type, constraint_type);
    if (reconstruction_space == VectorTaylorPolynomial) {
        return dimension*added_alpha_size;
    } else {
        return added_alpha_size;
    }
}

KOKKOS_INLINE_FUNCTION
int getRHSSquareDim(DenseSolverType dense_solver_type, ConstraintType constraint_type, ReconstructionSpace reconstruction_space, const int dimension, const int M, const int N) {
    // Return the appropriate size for _RHS. Since in LU, the system solves P^T*P against P^T*W.
    // We store P^T*P in the RHS space, which means RHS can be much smaller compared to the
    // case for QR/SVD where the system solves PsqrtW against sqrtW*Identity

    int added_coeff_size = getAdditionalCoeffSizeFromConstraintAndSpace(dense_solver_type, constraint_type, reconstruction_space, dimension);

    if (constraint_type == NEUMANN_GRAD_SCALAR) {
        return N + added_coeff_size;
    } else {
        if (dense_solver_type != LU) {
            return M;
        } else {
            return N + added_coeff_size;
        }
    }
}

KOKKOS_INLINE_FUNCTION
void getPDims(DenseSolverType dense_solver_type, ConstraintType constraint_type, ReconstructionSpace reconstruction_space, const int dimension, const int M, const int N, int &out_row, int &out_col) {
    // Return the appropriate size for _P.
    // In the case of solving with LU and additional constraint is used, _P needs
    // to be resized to include additional row(s) based on the type of constraint.

    int added_coeff_size = getAdditionalCoeffSizeFromConstraintAndSpace(dense_solver_type, constraint_type, reconstruction_space, dimension);
    int added_alpha_size = getAdditionalAlphaSizeFromConstraint(dense_solver_type, constraint_type);

    if (constraint_type == NEUMANN_GRAD_SCALAR) {
        out_row = M + added_alpha_size;
        out_col = N + added_coeff_size;
    } else {
        if (dense_solver_type == LU) {
            out_row = M + added_alpha_size;
            out_col = N + added_coeff_size;
        } else {
            out_row = M;
            out_col = N;
        }
    }
}

} // Compadre

#endif
