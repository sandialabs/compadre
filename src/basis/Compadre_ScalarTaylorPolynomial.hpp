#ifndef _COMPADRE_SCALAR_TAYLORPOLYNOMIAL_BASIS_HPP_
#define _COMPADRE_SCALAR_TAYLORPOLYNOMIAL_BASIS_HPP_

#include "Compadre_GMLS.hpp"

namespace Compadre {

    const double factorial[15] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};

    // for order P, the sum of the basis is defined by
    //
    // (in 1D) \sum_{n=0}^{n=P} (x/h)^n / n!
    // (in 2D) \sum_{n=0}^{n=P} \sum_{k=0}^{k=n} (x/h)^(n-k)*(y/h)^k / ((n-k)!k!)
    // (in 3D) \sum_{p=0}^{p=P} \sum_{k1+k2+k3=n} (x/h)^k1*(y/h)^k2*(z/h)^k3 / (k1!k2!k3!)
    //
    
    KOKKOS_INLINE_FUNCTION
    int getSize(const int degree, const int dimension) {
        if (dimension == 3) return (degree+1)*(degree+2)*(degree+3)/6;
        else if (dimension == 2) return (degree+1)*(degree+2)/2;
        else return degree+1;
    }

    KOKKOS_INLINE_FUNCTION
    void calcScalarTaylorBasis(double* delta, const int max_degree, const double h, const double x) {
        double x_over_h_to_i[max_degree];
        x_over_h_to_i[0] = 1;
        for (int i=1; i<=max_degree; ++i) {
            x_over_h_to_i[i] = x_over_h_to_i[i-1]*(x/h);
        }
        for (int i=0; i<=max_degree; ++i) {
            *(delta+i) = x_over_h_to_i[i]/factorial[i];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void calcScalarTaylorBasis(double* delta, const int max_degree, const double x, const double y) {
        double x_over_h_to_i[max_degree];
        double y_over_h_to_i[max_degree];
        x_over_h_to_i[0] = 1;
        y_over_h_to_i[0] = 1;
        for (int i=1; i<=max_degree; ++i) {
            x_over_h_to_i[i] = x_over_h_to_i[i-1]*(x/h);
            y_over_h_to_i[i] = y_over_h_to_i[i-1]*(y/h);
        }
        // (in 2D) \sum_{n=0}^{n=P} \sum_{k=0}^{k=n} (x/h)^(n-k)*(y/h)^k / ((n-k)!k!)
        int alphax, alphay, alphaz;
        double alphaf;
        int i = 0;
        for (int n = 0; n <= max_degree; n++){
            for (alphay = 0; alphay <= n; alphay++){
                alphax = n - alphay;
                alphaf = factorial[alphax]*factorial[alphay];
                *(delta+i) = x_over_h_to_i[alphax]*y_over_h_to_i[alphay]/alphaf;
                i++;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void calcScalarTaylorBasisVectorLevel(double* delta, const int max_degree, const double x, const double y, const double z) {
        double x_over_h_to_i[max_degree];
        double y_over_h_to_i[max_degree];
        double z_over_h_to_i[max_degree];
        x_over_h_to_i[0] = 1;
        y_over_h_to_i[0] = 1;
        z_over_h_to_i[0] = 1;
        for (int i=1; i<=max_degree; ++i) {
            x_over_h_to_i[i] = x_over_h_to_i[i-1]*(x/h);
            y_over_h_to_i[i] = y_over_h_to_i[i-1]*(y/h);
            z_over_h_to_i[i] = z_over_h_to_i[i-1]*(z/h);
        }
        // (in 3D) \sum_{p=0}^{p=P} \sum_{k1+k2+k3=n} (x/h)^k1*(y/h)^k2*(z/h)^k3 / (k1!k2!k3!)
        int alphax, alphay, alphaz;
        double alphaf;
        int i=0, s=0;
        for (int n = 0; n <= max_degree; n++){
            for (alphaz = 0; alphaz <= n; alphaz++){
                s = n - alphaz;
                for (alphay = 0; alphay <= s; alphay++){
                    alphax = s - alphay;
                    alphaf = factorial[alphax]*factorial[alphay]*factorial[alphaz];
                    *(delta+i) = x_over_h_to_i[alphax]*y_over_h_to_i[alphay]*z_over_h_to_i[alphaz]/alphaf;
                    i++;
                }
            }
        }
    }
}

#endif
