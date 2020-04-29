#ifndef _COMPADRE_GMLS_DIVFREE_BASIS_HPP_
#define _COMPADRE_GMLS_DIVFREE_BASIS_HPP_

#include "Compadre_GMLS.hpp"

namespace Compadre {
namespace DivergenceFreePolynomialBasis {
    // Calculating the basis functions
    KOKKOS_INLINE_FUNCTION
    XYZ evaluate(int np, const double sx, const double sy, const double sz) {
        // define one-third
        const double th = 1./3.;

        switch (np) {
            // P0
            case 0: return XYZ(1.0,0.0,0.0);
            case 1: return XYZ(0.0,1.0,0.0);
            case 2: return XYZ(0.0,0.0,1.0);

            // P1
            case 3 : return XYZ( sy,  0,  0);
            case 4 : return XYZ( sz,  0,  0);
            case 5 : return XYZ(  0, sx,  0);
            case 6 : return XYZ(  0, sz,  0);
            case 7 : return XYZ(  0,  0, sx);
            case 8 : return XYZ(  0,  0, sy);
            case 9 : return XYZ( sx,-sy,  0);
            case 10: return XYZ( sx,  0,-sz);

            // P2
            case 11: return XYZ(     sy*sy,          0,         0);
            case 12: return XYZ(     sz*sz,          0,         0);
            case 13: return XYZ(     sy*sz,          0,         0);
            case 14: return XYZ(         0,      sx*sx,         0);
            case 15: return XYZ(         0,      sz*sz,         0);
            case 16: return XYZ(         0,      sx*sz,         0);
            case 17: return XYZ(         0,          0,     sx*sx);
            case 18: return XYZ(         0,          0,     sy*sy);
            case 19: return XYZ(         0,          0,     sx*sy);

            case 20: return XYZ(     sx*sx, -2.0*sx*sy,         0);
            case 21: return XYZ(     sx*sy,          0,-1.0*sy*sz);
            case 22: return XYZ(     sx*sz, -1.0*sy*sz,         0);
            case 23: return XYZ(-2.0*sx*sy,      sy*sy,          0);
            case 24: return XYZ(         0,      sx*sy, -1.0*sx*sz);
            case 25: return XYZ(-2.0*sx*sz,          0,      sz*sz);

            // P3
            case 26: return XYZ(sy*sy*sy   ,            0,             0);
            case 27: return XYZ(sy*sy*sz   ,            0,             0);
            case 28: return XYZ(sy*sz*sz   ,            0,             0);
            case 29: return XYZ(sz*sz*sz   ,            0,             0);
            case 30: return XYZ(          0,  sx*sx*sx   ,             0);
            case 31: return XYZ(          0,  sx*sx*sz   ,             0);
            case 32: return XYZ(          0,  sx*sz*sz   ,             0);
            case 33: return XYZ(          0,  sz*sz*sz   ,             0);
            case 34: return XYZ(          0,            0,  sx*sx*sx    );
            case 35: return XYZ(          0,            0,  sx*sx*sy    );
            case 36: return XYZ(          0,            0,  sx*sy*sy    );
            case 37: return XYZ(          0,            0,  sy*sy*sy    );

            case 38: return XYZ( sx*sx*sx  ,-3.0*sx*sx*sy,            0);
            case 39: return XYZ( sx*sx*sx  ,            0,-3.0*sx*sx*sz);
            case 40: return XYZ( sx*sx*sy  ,-1.0*sx*sy*sy,            0);
            case 41: return XYZ( sx*sx*sy  ,            0,-2.0*sx*sy*sz);
            case 42: return XYZ( sx*sx*sz  ,-2.0*sx*sy*sz,            0);
            case 43: return XYZ( sx*sx*sz  ,            0,-1.0*sx*sz*sz);
            case 44: return XYZ( sx*sy*sy  ,- th*sy*sy*sy,            0);
            case 45: return XYZ( sx*sy*sy  ,            0,-1.0*sy*sy*sz);
            case 46: return XYZ( sx*sy*sz  ,-0.5*sy*sy*sz,            0);
            case 47: return XYZ( sx*sy*sz  ,            0,-0.5*sy*sz*sz);
            case 48: return XYZ( sx*sz*sz  ,-1.0*sy*sz*sz,            0);
            case 49: return XYZ( sx*sz*sz  ,            0,- th*sz*sz*sz);

            // P4
            case 50: return XYZ(sy*sy*sy*sy         ,                   0,                   0);
            case 51: return XYZ(sy*sy*sy*sz         ,                   0,                   0);
            case 52: return XYZ(sy*sy*sz*sz         ,                   0,                   0);
            case 53: return XYZ(sy*sz*sz*sz         ,                   0,                   0);
            case 54: return XYZ(sz*sz*sz*sz         ,                   0,                   0);
            case 55: return XYZ(                   0, sx*sx*sx*sx        ,                   0);
            case 56: return XYZ(                   0, sx*sx*sx*sz        ,                   0);
            case 57: return XYZ(                   0, sx*sx*sz*sz        ,                   0);
            case 58: return XYZ(                   0, sx*sz*sz*sz        ,                   0);
            case 59: return XYZ(                   0, sz*sz*sz*sz        ,                   0);
            case 60: return XYZ(                   0,                   0, sx*sx*sx*sx        );
            case 61: return XYZ(                   0,                   0, sx*sx*sx*sy        );
            case 62: return XYZ(                   0,                   0, sx*sx*sy*sy        );
            case 63: return XYZ(                   0,                   0, sx*sy*sy*sy        );
            case 64: return XYZ(                   0,                   0, sy*sy*sy*sy        );

            case 65: return XYZ(sx*sx*sx*sx         ,-3.0*sx*sx*sx*sy    ,                   0);
            case 66: return XYZ(sx*sx*sx*sx         ,                   0, -4.0*sx*sx*sx*sz   );
            case 67: return XYZ(sx*sx*sx*sy         ,-1.5*sx*sx*sy*sy    ,                   0);
            case 68: return XYZ(sx*sx*sx*sy         ,                   0, -3.0*sx*sx*sy*sz   );
            case 69: return XYZ(sx*sx*sx*sz         ,-3.0*sx*sx*sy*sz    ,                   0);
            case 70: return XYZ(sx*sx*sx*sz         ,                   0, -1.5*sx*sx*sz*sz   );
            case 71: return XYZ(sx*sx*sy*sy         , -2.0*th*sx*sy*sy*sy,                   0);
            case 72: return XYZ(sx*sx*sy*sy         ,                   0, -2.0*sx*sy*sy*sz   );
            case 73: return XYZ(sx*sx*sy*sz         , -sx*sy*sy*sz       ,                   0);
            case 74: return XYZ(sx*sx*sy*sz         ,                   0, -sx*sy*sz*sz       );
            case 75: return XYZ(sx*sx*sz*sz         , -2.0*sx*sy*sz*sz   ,                   0);
            case 76: return XYZ(sx*sx*sz*sz         ,                   0, -2.0*th*sx*sz*sz*sz);
            case 77: return XYZ(sx*sy*sy*sy         , -0.25*sy*sy*sy*sy  ,                   0);
            case 78: return XYZ(sx*sy*sy*sy         ,                   0, -sy*sy*sy*sz       );
            case 79: return XYZ(sx*sy*sy*sz         , -th*sy*sy*sy*sz    ,                   0);
            case 80: return XYZ(sx*sy*sy*sz         ,                   0, -0.5*sy*sy*sz*sz   );
            case 81: return XYZ(sx*sy*sz*sz         , -0.5*sy*sy*sz*sz   ,                   0);
            case 82: return XYZ(sx*sy*sz*sz         ,                   0, -th*sy*sz*sz*sz    );
            case 83: return XYZ(sx*sz*sz*sz         , -sy*sz*sz*sz       ,                   0);
            case 84: return XYZ(sx*sz*sz*sz         ,                   0,-0.25*sz*sz*sz*sz   );

            // default
            default: compadre_kernel_assert_release((false) && "Divergence-free basis only sup\
ports up to 4th-order polynomials for now.");
                     return XYZ(0,0,0); // avoid warning about no return
        }
    }

    KOKKOS_INLINE_FUNCTION
    XYZ evaluate(int np, const double sx, const double sy) {
        // define one-third

        switch (np) {
            // P0
            case 0: return XYZ(1.0,0.0,0.0);
            case 1: return XYZ(0.0,1.0,0.0);

            // P1
            case 2: return XYZ(sy,  0, 0);
            case 3: return XYZ( 0, sx, 0);
            case 4: return XYZ(sx,-sy, 0);

            // P2
            case 5: return XYZ(      sy*sy,          0, 0);
            case 6: return XYZ(          0,      sx*sx, 0);

            case 7: return XYZ(      sx*sx, -2.0*sx*sy, 0);
            case 8: return XYZ( -2.0*sx*sy,      sy*sy, 0);

            // P3
            case 9 : return XYZ(     sy*sy*sy,             0, 0);
            case 10: return XYZ(            0,      sx*sx*sx, 0);

            case 11: return XYZ(     sx*sx*sx, -3.0*sx*sx*sy, 0);
            case 12: return XYZ(     sx*sx*sy, -1.0*sx*sy*sy, 0);
            case 13: return XYZ(-3.0*sx*sy*sy,      sy*sy*sy, 0);

            // P4
            case 14: return XYZ(     sy*sy*sy*sy,                0, 0);
            case 15: return XYZ(               0,      sx*sx*sx*sx, 0);

            case 16: return XYZ(     sx*sx*sx*sx, -4.0*sx*sx*sx*sy, 0);
            case 17: return XYZ( 2.0*sx*sx*sx*sy, -3.0*sx*sx*sy*sy, 0);
            case 18: return XYZ(-3.0*sx*sx*sy*sy,  2.0*sx*sy*sy*sy, 0);
            case 19: return XYZ(-4.0*sx*sy*sy*sy,      sy*sy*sy*sy, 0);

        // default
            default: compadre_kernel_assert_release((false) && "Divergence-free basis only sup\
ports up to 4th-order polynomials for now.");
                     return XYZ(0,0,0); // avoid warning about no return
        }
    }

    // Calculating the basis functions
    KOKKOS_INLINE_FUNCTION
    XYZ evaluatePartialDerivative(int np, const int partial_direction, const double h, const double sx, const double sy, const double sz) {
        // define one-third
        const double th = 1./3.;

        if (partial_direction==0) {
            switch (np) {
                // P0
                case 0: return XYZ(0.0,0.0,0.0);
                case 1: return XYZ(0.0,0.0,0.0);
                case 2: return XYZ(0.0,0.0,0.0);

                // P1
                case 3 : return XYZ(      0,      0,      0);
                case 4 : return XYZ(      0,      0,      0);
                case 5 : return XYZ(      0,  1.0/h,      0);
                case 6 : return XYZ(      0,      0,      0);
                case 7 : return XYZ(      0,      0,  1.0/h);
                case 8 : return XYZ(      0,      0,      0);
                case 9 : return XYZ(  1.0/h,      0,      0);
                case 10: return XYZ(  1.0/h,      0,      0);

                // P2
                case 11: return XYZ(         0,          0,         0);
                case 12: return XYZ(         0,          0,         0);
                case 13: return XYZ(         0,          0,         0);
                case 14: return XYZ(         0,     2*sx/h,         0);
                case 15: return XYZ(         0,          0,         0);
                case 16: return XYZ(         0,       sz/h,         0);
                case 17: return XYZ(         0,          0,    2*sx/h);
                case 18: return XYZ(         0,          0,         0);
                case 19: return XYZ(         0,          0,      sy/h);

                case 20: return XYZ(    2*sx/h,  -2.0*sy/h,         0);
                case 21: return XYZ(      sy/h,          0,         0);
                case 22: return XYZ(      sz/h,          0,         0);
                case 23: return XYZ( -2.0*sy/h,          0,         0);
                case 24: return XYZ(         0,       sy/h, -1.0*sz/h);
                case 25: return XYZ( -2.0*sz/h,          0,         0);

                // P3
                case 26: return XYZ(          0,            0,             0);
                case 27: return XYZ(          0,            0,             0);
                case 28: return XYZ(          0,            0,             0);
                case 29: return XYZ(          0,            0,             0);
                case 30: return XYZ(          0,    3*sx*sx/h,             0);
                case 31: return XYZ(          0,    2*sx*sz/h,             0);
                case 32: return XYZ(          0,      sz*sz/h,             0);
                case 33: return XYZ(          0,            0,             0);
                case 34: return XYZ(          0,            0,     3*sx*sx/h);
                case 35: return XYZ(          0,            0,     2*sx*sy/h);
                case 36: return XYZ(          0,            0,       sy*sy/h);
                case 37: return XYZ(          0,            0,             0);

                case 38: return XYZ(  3*sx*sx/h, -6.0*sx*sy/h,            0);
                case 39: return XYZ(  3*sx*sx/h,            0, -6.0*sx*sz/h);
                case 40: return XYZ(  2*sx*sy/h, -1.0*sy*sy/h,            0);
                case 41: return XYZ(  2*sx*sy/h,            0, -2.0*sy*sz/h);
                case 42: return XYZ(  2*sx*sz/h, -2.0*sy*sz/h,            0);
                case 43: return XYZ(  2*sx*sz/h,            0, -1.0*sz*sz/h);
                case 44: return XYZ(    sy*sy/h,            0,            0);
                case 45: return XYZ(    sy*sy/h,            0,            0);
                case 46: return XYZ(    sy*sz/h,            0,            0);
                case 47: return XYZ(    sy*sz/h,            0,            0);
                case 48: return XYZ(    sz*sz/h,            0,            0);
                case 49: return XYZ(    sz*sz/h,            0,            0);

                // P4
                case 50: return XYZ(                   0,                   0,                   0);
                case 51: return XYZ(                   0,                   0,                   0);
                case 52: return XYZ(                   0,                   0,                   0);
                case 53: return XYZ(                   0,                   0,                   0);
                case 54: return XYZ(                   0,                   0,                   0);
                case 55: return XYZ(                   0,        4*sx*sx*sx/h,                   0);
                case 56: return XYZ(                   0,        3*sx*sx*sz/h,                   0);
                case 57: return XYZ(                   0,        2*sx*sz*sz/h,                   0);
                case 58: return XYZ(                   0,          sz*sz*sz/h,                   0);
                case 59: return XYZ(                   0,                   0,                   0);
                case 60: return XYZ(                   0,                   0,        4*sx*sx*sx/h);
                case 61: return XYZ(                   0,                   0,        3*sx*sx*sy/h);
                case 62: return XYZ(                   0,                   0,        2*sx*sy*sy/h);
                case 63: return XYZ(                   0,                   0,          sy*sy*sy/h);
                case 64: return XYZ(                   0,                   0,                   0);

                case 65: return XYZ( 4*sx*sx*sx/h       ,-9.0*sx*sx*sy/h    ,                   0);
                case 66: return XYZ( 4*sx*sx*sx/h       ,                   0, -12.0*sx*sx*sz/h   );
                case 67: return XYZ( 3*sx*sx*sy/h       ,-3.0*sx*sy*sy/h    ,                   0);
                case 68: return XYZ( 3*sx*sx*sy/h       ,                   0, -6.0*sx*sy*sz/h   );
                case 69: return XYZ( 3*sx*sx*sz/h       ,-6.0*sx*sy*sz/h    ,                   0);
                case 70: return XYZ( 3*sx*sx*sz/h       ,                   0, -3.0*sx*sz*sz/h   );
                case 71: return XYZ( 2*sx*sy*sy/h       , -2.0*th*sy*sy*sy/h,                   0);
                case 72: return XYZ( 2*sx*sy*sy/h       ,                   0, -2.0*sy*sy*sz/h   );
                case 73: return XYZ( 2*sx*sy*sz/h       , -sy*sy*sz/h       ,                   0);
                case 74: return XYZ( 2*sx*sy*sz/h       ,                   0, -sy*sz*sz/h       );
                case 75: return XYZ( 2*sx*sz*sz/h       , -2.0*sy*sz*sz/h   ,                   0);
                case 76: return XYZ( 2*sx*sz*sz/h       ,                   0, -2.0*th*sz*sz*sz/h);
                case 77: return XYZ(   sy*sy*sy/h       ,                   0,                   0);
                case 78: return XYZ(   sy*sy*sy/h       ,                   0,                   0);
                case 79: return XYZ(   sy*sy*sz/h       ,                   0,                   0);
                case 80: return XYZ(   sy*sy*sz/h       ,                   0,                   0);
                case 81: return XYZ(   sy*sz*sz/h       ,                   0,                   0);
                case 82: return XYZ(   sy*sz*sz/h       ,                   0,                   0);
                case 83: return XYZ(   sz*sz*sz/h       ,                   0,                   0);
                case 84: return XYZ(   sz*sz*sz/h       ,                   0,                   0);

                // default
                default: compadre_kernel_assert_release((false) && "Divergence-free basis only sup\
ports up     to 4th-order polynomials for now.");
                         return XYZ(0,0,0); // avoid warning about no return
            }
        } else if (partial_direction==1) {
            switch (np) {
                // P0
                case 0: return XYZ(0.0,0.0,0.0);
                case 1: return XYZ(0.0,0.0,0.0);
                case 2: return XYZ(0.0,0.0,0.0);

                // P1
                case 3 : return XYZ( 1.0/h,      0,     0);
                case 4 : return XYZ(     0,      0,     0);
                case 5 : return XYZ(     0,      0,     0);
                case 6 : return XYZ(     0,      0,     0);
                case 7 : return XYZ(     0,      0,     0);
                case 8 : return XYZ(     0,      0, 1.0/h);
                case 9 : return XYZ(     0, -1.0/h,     0);
                case 10: return XYZ(     0,      0,     0);

                // P2
                case 11: return XYZ(    2*sy/h,          0,          0);
                case 12: return XYZ(         0,          0,          0);
                case 13: return XYZ(      sz/h,          0,          0);
                case 14: return XYZ(         0,          0,          0);
                case 15: return XYZ(         0,          0,          0);
                case 16: return XYZ(         0,          0,          0);
                case 17: return XYZ(         0,          0,          0);
                case 18: return XYZ(         0,          0,     2*sy/h);
                case 19: return XYZ(         0,          0,       sx/h);

                case 20: return XYZ(         0,  -2.0*sx/h,          0);
                case 21: return XYZ(      sx/h,          0,  -1.0*sz/h);
                case 22: return XYZ(         0,  -1.0*sz/h,          0);
                case 23: return XYZ( -2.0*sx/h,     2*sy/h,          0);
                case 24: return XYZ(         0,       sx/h,          0);
                case 25: return XYZ(         0,          0,          0);

                // P3
                case 26: return XYZ(  3*sy*sy/h,            0,             0);
                case 27: return XYZ(  2*sy*sz/h,            0,             0);
                case 28: return XYZ(    sz*sz/h,            0,             0);
                case 29: return XYZ(          0,            0,             0);
                case 30: return XYZ(          0,            0,             0);
                case 31: return XYZ(          0,            0,             0);
                case 32: return XYZ(          0,            0,             0);
                case 33: return XYZ(          0,            0,             0);
                case 34: return XYZ(          0,            0,             0);
                case 35: return XYZ(          0,            0,       sx*sx/h);
                case 36: return XYZ(          0,            0,     2*sx*sy/h);
                case 37: return XYZ(          0,            0,     3*sy*sy/h);

                case 38: return XYZ(          0, -3.0*sx*sx/h,            0);
                case 39: return XYZ(          0,            0,            0);
                case 40: return XYZ(    sx*sx/h, -2.0*sx*sy/h,            0);
                case 41: return XYZ(    sx*sx/h,            0, -2.0*sx*sz/h);
                case 42: return XYZ(          0, -2.0*sx*sz/h,            0);
                case 43: return XYZ(          0,            0,            0);
                case 44: return XYZ(  2*sx*sy/h,-3*th*sy*sy/h,            0);
                case 45: return XYZ(  2*sx*sy/h,            0, -2.0*sy*sz/h);
                case 46: return XYZ(    sx*sz/h, -1.0*sy*sz/h,            0);
                case 47: return XYZ(    sx*sz/h,            0, -0.5*sz*sz/h);
                case 48: return XYZ(          0, -1.0*sz*sz/h,            0);
                case 49: return XYZ(          0,            0,            0);

                // P4
                case 50: return XYZ(        4*sy*sy*sy/h,                   0,                   0);
                case 51: return XYZ(        3*sy*sy*sz/h,                   0,                   0);
                case 52: return XYZ(        2*sy*sz*sz/h,                   0,                   0);
                case 53: return XYZ(          sz*sz*sz/h,                   0,                   0);
                case 54: return XYZ(                   0,                   0,                   0);
                case 55: return XYZ(                   0,                   0,                   0);
                case 56: return XYZ(                   0,                   0,                   0);
                case 57: return XYZ(                   0,                   0,                   0);
                case 58: return XYZ(                   0,                   0,                   0);
                case 59: return XYZ(                   0,                   0,                   0);
                case 60: return XYZ(                   0,                   0,                   0);
                case 61: return XYZ(                   0,                   0,         sx*sx*sx/h);
                case 62: return XYZ(                   0,                   0,         2*sx*sx*sy/h);
                case 63: return XYZ(                   0,                   0,         3*sx*sy*sy/h);
                case 64: return XYZ(                   0,                   0,         4*sy*sy*sy/h);

                case 65: return XYZ(                   0,     -3.0*sx*sx*sx/h,                   0);
                case 66: return XYZ(                   0,                   0,                   0);
                case 67: return XYZ(          sx*sx*sx/h,     -3.0*sx*sx*sy/h,                   0);
                case 68: return XYZ(          sx*sx*sx/h,                   0,     -3.0*sx*sx*sz/h);
                case 69: return XYZ(                   0,     -3.0*sx*sx*sz/h,                   0);
                case 70: return XYZ(                   0,                   0,                   0);
                case 71: return XYZ(        2*sx*sx*sy/h,  -6.0*th*sx*sy*sy/h,                   0);
                case 72: return XYZ(        2*sx*sx*sy/h,                   0,     -4.0*sx*sy*sz/h);
                case 73: return XYZ(          sx*sx*sz/h,         -2*sx*sy*sz,                   0);
                case 74: return XYZ(          sx*sx*sz/h,                   0,         -sx*sz*sz/h);
                case 75: return XYZ(                   0,     -2.0*sx*sz*sz/h,                   0);
                case 76: return XYZ(                   0,                   0,                   0);
                case 77: return XYZ(        3*sx*sy*sy/h,     -1.0*sy*sy*sy/h,                   0);
                case 78: return XYZ(        3*sx*sy*sy/h,                   0,       -3*sy*sy*sz/h);
                case 79: return XYZ(        2*sx*sy*sz/h,    -3*th*sy*sy*sz/h,                   0);
                case 80: return XYZ(        2*sx*sy*sz/h,                   0,     -1.0*sy*sz*sz/h);
                case 81: return XYZ(          sx*sz*sz/h,     -1.0*sy*sz*sz/h,                   0);
                case 82: return XYZ(          sx*sz*sz/h,                   0,      -th*sz*sz*sz/h);
                case 83: return XYZ(                   0,         -sz*sz*sz/h,                   0);
                case 84: return XYZ(                   0,                   0,                   0);

                // default
                default: compadre_kernel_assert_release((false) && "Divergence-free basis only sup\
ports up     to 4th-order polynomials for now.");
                         return XYZ(0,0,0); // avoid warning about no return
            }
        } else {
            switch (np) {
                // P0
                case 0: return XYZ(0.0,0.0,0.0);
                case 1: return XYZ(0.0,0.0,0.0);
                case 2: return XYZ(0.0,0.0,0.0);

                // P1
                case 3 : return XYZ(   0,   0,   0);
                case 4 : return XYZ( 1/h,   0,   0);
                case 5 : return XYZ(   0,   0,   0);
                case 6 : return XYZ(   0, 1/h,   0);
                case 7 : return XYZ(   0,   0,   0);
                case 8 : return XYZ(   0,   0,   0);
                case 9 : return XYZ(   0,   0,   0);
                case 10: return XYZ(   0,   0,-1/h);

                // P2
                case 11: return XYZ(         0,          0,         0);
                case 12: return XYZ(    2*sz/h,          0,         0);
                case 13: return XYZ(      sy/h,          0,         0);
                case 14: return XYZ(         0,          0,         0);
                case 15: return XYZ(         0,     2*sz/h,         0);
                case 16: return XYZ(         0,       sx/h,         0);
                case 17: return XYZ(         0,          0,         0);
                case 18: return XYZ(         0,          0,         0);
                case 19: return XYZ(         0,          0,         0);

                case 20: return XYZ(         0,          0,         0);
                case 21: return XYZ(         0,          0, -1.0*sy/h);
                case 22: return XYZ(      sx/h,  -1.0*sy/h,         0);
                case 23: return XYZ(         0,          0,         0);
                case 24: return XYZ(         0,          0, -1.0*sx/h);
                case 25: return XYZ( -2.0*sx/h,          0,    2*sz/h);

                // P3
                case 26: return XYZ(          0,            0,             0);
                case 27: return XYZ(    sy*sy/h,            0,             0);
                case 28: return XYZ(  2*sy*sz/h,            0,             0);
                case 29: return XYZ(  3*sz*sz/h,            0,             0);
                case 30: return XYZ(          0,            0,             0);
                case 31: return XYZ(          0,      sx*sx/h,             0);
                case 32: return XYZ(          0,    2*sx*sz/h,             0);
                case 33: return XYZ(          0,    3*sz*sz/h,             0);
                case 34: return XYZ(          0,            0,             0);
                case 35: return XYZ(          0,            0,             0);
                case 36: return XYZ(          0,            0,             0);
                case 37: return XYZ(          0,            0,             0);

                case 38: return XYZ(          0,            0,            0);
                case 39: return XYZ(          0,            0, -3.0*sx*sx/h);
                case 40: return XYZ(          0,            0,            0);
                case 41: return XYZ(          0,            0, -2.0*sx*sy/h);
                case 42: return XYZ(    sx*sx/h, -2.0*sx*sy/h,            0);
                case 43: return XYZ(    sx*sx/h,            0, -2.0*sx*sz/h);
                case 44: return XYZ(          0,            0,            0);
                case 45: return XYZ(          0,            0, -1.0*sy*sy/h);
                case 46: return XYZ(    sx*sy/h, -0.5*sy*sy/h,            0);
                case 47: return XYZ(    sx*sy/h,            0, -1.0*sy*sz/h);
                case 48: return XYZ(  2*sx*sz/h, -2.0*sy*sz/h,            0);
                case 49: return XYZ(  2*sx*sz/h,            0,-3*th*sz*sz/h);

                // P4
                case 50: return XYZ(                   0,                   0,                   0);
                case 51: return XYZ(          sy*sy*sy/h,                   0,                   0);
                case 52: return XYZ(        2*sy*sy*sz/h,                   0,                   0);
                case 53: return XYZ(        3*sy*sz*sz/h,                   0,                   0);
                case 54: return XYZ(        4*sz*sz*sz/h,                   0,                   0);
                case 55: return XYZ(                   0,                   0,                   0);
                case 56: return XYZ(                   0,          sx*sx*sx/h,                   0);
                case 57: return XYZ(                   0,        2*sx*sx*sz/h,                   0);
                case 58: return XYZ(                   0,        3*sx*sz*sz/h,                   0);
                case 59: return XYZ(                   0,        4*sz*sz*sz/h,                   0);
                case 60: return XYZ(                   0,                   0,                   0);
                case 61: return XYZ(                   0,                   0,                   0);
                case 62: return XYZ(                   0,                   0,                   0);
                case 63: return XYZ(                   0,                   0,                   0);
                case 64: return XYZ(                   0,                   0,                   0);

                case 65: return XYZ(                   0,                   0,                   0);
                case 66: return XYZ(                   0,                   0,     -4.0*sx*sx*sx/h);
                case 67: return XYZ(                   0,                   0,                   0);
                case 68: return XYZ(                   0,                   0,     -3.0*sx*sx*sy/h);
                case 69: return XYZ(          sx*sx*sx/h,     -3.0*sx*sx*sy/h,                   0);
                case 70: return XYZ(          sx*sx*sx/h,                   0,     -3.0*sx*sx*sz/h);
                case 71: return XYZ(                   0,                   0,                   0);
                case 72: return XYZ(                   0,                   0,     -2.0*sx*sy*sy/h);
                case 73: return XYZ(          sx*sx*sy/h,         -sx*sy*sy/h,                   0);
                case 74: return XYZ(          sx*sx*sy/h,                   0,       -2*sx*sy*sz/h);
                case 75: return XYZ(        2*sx*sx*sz/h,     -4.0*sx*sy*sz/h,                   0);
                case 76: return XYZ(        2*sx*sx*sz/h,                   0,  -6.0*th*sx*sz*sz/h);
                case 77: return XYZ(                   0,                   0,                   0);
                case 78: return XYZ(                   0,                   0,         -sy*sy*sy/h);
                case 79: return XYZ(          sx*sy*sy/h,      -th*sy*sy*sy/h,                   0);
                case 80: return XYZ(          sx*sy*sy/h,                   0,     -1.0*sy*sy*sz/h);
                case 81: return XYZ(        2*sx*sy*sz/h,     -1.0*sy*sy*sz/h,                   0);
                case 82: return XYZ(        2*sx*sy*sz/h,                   0,    -3*th*sy*sz*sz/h);
                case 83: return XYZ(        3*sx*sz*sz/h,       -3*sy*sz*sz/h,                   0);
                case 84: return XYZ(        3*sx*sz*sz/h,                   0,     -1.0*sz*sz*sz/h);

                // default
                default: compadre_kernel_assert_release((false) && "Divergence-free basis only sup\
ports up     to 4th-order polynomials for now.");
                         return XYZ(0,0,0); // avoid warning about no return
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    XYZ evaluatePartialDerivative(int np, const int partial_direction, const double h, const double sx, const double sy) {
        // define one-third

        if (partial_direction==0) {
            switch (np) {
                // P0
                case 0: return XYZ(0.0,0.0,0.0);
                case 1: return XYZ(0.0,0.0,0.0);

                // P1
                case 2: return XYZ(     0,      0, 0);
                case 3: return XYZ(     0,  1.0/h, 0);
                case 4: return XYZ( 1.0/h,      0, 0);

                // P2
                case 5: return XYZ(          0,            0, 0);
                case 6: return XYZ(          0,       2*sx/h, 0);

                case 7: return XYZ(       2*sx/h,    -2.0*sy/h, 0);
                case 8: return XYZ(    -2.0*sy/h,            0, 0);

                // P3
                case 9 : return XYZ(            0,               0, 0);
                case 10: return XYZ(            0,       3*sx*sx/h, 0);

                case 11: return XYZ(      3*sx*sx/h,    -6.0*sx*sy/h, 0);
                case 12: return XYZ(    2.0*sx*sy/h,    -1.0*sy*sy/h, 0);
                case 13: return XYZ(   -3.0*sy*sy/h,               0, 0);

                // P4
                case 14: return XYZ(               0,                  0, 0);
                case 15: return XYZ(               0,       4*sx*sx*sx/h, 0);

                case 16: return XYZ(    4*sx*sx*sx/h,   -12.0*sx*sx*sy/h, 0);
                case 17: return XYZ(  6.0*sx*sx*sy/h,    -6.0*sx*sy*sy/h, 0);
                case 18: return XYZ( -6.0*sx*sy*sy/h,     2.0*sy*sy*sy/h, 0);
                case 19: return XYZ( -4.0*sy*sy*sy/h,                  0, 0);

            // default
                default: compadre_kernel_assert_release((false) && "Divergence-free basis only sup\
ports up     to 4th-order polynomials for now.");
                         return XYZ(0,0,0); // avoid warning about no return
            }
        } else {
            switch (np) {
                // P0
                case 0: return XYZ(0.0,0.0,0.0);
                case 1: return XYZ(0.0,0.0,0.0);

                // P1
                case 2: return XYZ( 1.0/h,      0, 0);
                case 3: return XYZ(     0,      0, 0);
                case 4: return XYZ(     0, -1.0/h, 0);

                // P2
                case 5: return XYZ(       2*sy/h,          0, 0);
                case 6: return XYZ(            0,          0, 0);

                case 7: return XYZ(            0,    -2.0*sx/h, 0);
                case 8: return XYZ(    -2.0*sx/h,       2*sy/h, 0);

                // P3
                case 9 : return XYZ(      3*sy*sy/h,             0, 0);
                case 10: return XYZ(              0,             0, 0);

                case 11: return XYZ(              0,    -3.0*sx*sx/h, 0);
                case 12: return XYZ(        sx*sx/h,    -2.0*sx*sy/h, 0);
                case 13: return XYZ(   -6.0*sx*sy/h,       3*sy*sy/h, 0);

                // P4
                case 14: return XYZ(      4*sy*sy*sy/h,                0, 0);
                case 15: return XYZ(                 0,                0, 0);

                case 16: return XYZ(                 0,    -4.0*sx*sx*sx/h, 0);
                case 17: return XYZ(    2.0*sx*sx*sx/h,    -6.0*sx*sx*sy/h, 0);
                case 18: return XYZ(   -6.0*sx*sx*sy/h,     6.0*sx*sy*sy/h, 0);
                case 19: return XYZ(  -12.0*sx*sy*sy/h,       4*sy*sy*sy/h, 0);

            // default
                default: compadre_kernel_assert_release((false) && "Divergence-free basis only sup\
ports up     to 4th-order polynomials for now.");
                         return XYZ(0,0,0); // avoid warning about no return
            }
        }
    }
} // DivergenceFreePolynomialBasis
} // Compadre

#endif
