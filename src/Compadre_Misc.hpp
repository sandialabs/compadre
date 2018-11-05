#ifndef _COMPADRE_MISC_HPP_
#define _COMPADRE_MISC_HPP_

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

}; // Compadre

#endif
