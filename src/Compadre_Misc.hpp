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
int getRHSDims(DenseSolverType dense_solver_type, ConstraintType constraint_type, const int M, const int N) {
    // Return the appropriate size for _RHS. Since in LU, the system solves P^T*P against P^T*W.
    // We store P^T*P in the RHS space, which means RHS can be much smaller compared to the
    // case for QR/SVD where the system solves PsqrtW against sqrtW*Identity
    if (dense_solver_type != LU) {
        return M;
    } else {
        return N;
    }
}

}; // Compadre

#endif
