#ifndef _COMPADRE_SCALAR_TAYLORPOLYNOMIAL_BASIS_HPP_
#define _COMPADRE_SCALAR_TAYLORPOLYNOMIAL_BASIS_HPP_

#include "Compadre_GMLS.hpp"

namespace Compadre {


namespace ScalarTaylorPolynomialBasis {


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
    void evaluate(const member_type& teamMember, double* delta, double* workspace, const int dimension, const int max_degree, const double h, const double x, const double y, const double z, const int starting_order = 0, const double weight_of_original_value = 0.0, const double weight_of_new_value = 1.0) {
        Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
            const double factorial[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};
            if (dimension==3) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                scratch_vector_type z_over_h_to_i(workspace+2*(max_degree+1), max_degree+1);
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
                for (int n = starting_order; n <= max_degree; n++){
                    for (alphaz = 0; alphaz <= n; alphaz++){
                        s = n - alphaz;
                        for (alphay = 0; alphay <= s; alphay++){
                            alphax = s - alphay;
                            alphaf = factorial[alphax]*factorial[alphay]*factorial[alphaz];
                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * x_over_h_to_i[alphax]*y_over_h_to_i[alphay]*z_over_h_to_i[alphaz]/alphaf;
                            i++;
                        }
                    }
                }
            } else if (dimension==2) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                x_over_h_to_i[0] = 1;
                y_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(x/h);
                    y_over_h_to_i[i] = y_over_h_to_i[i-1]*(y/h);
                }
                // (in 2D) \sum_{n=0}^{n=P} \sum_{k=0}^{k=n} (x/h)^(n-k)*(y/h)^k / ((n-k)!k!)
                int alphax, alphay;
                double alphaf;
                int i = 0;
                for (int n = starting_order; n <= max_degree; n++){
                    for (alphay = 0; alphay <= n; alphay++){
                        alphax = n - alphay;
                        alphaf = factorial[alphax]*factorial[alphay];
                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * x_over_h_to_i[alphax]*y_over_h_to_i[alphay]/alphaf;
                        i++;
                    }
                }
            } else {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                x_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(x/h);
                }
                // (in 1D) \sum_{n=0}^{n=P} (x/h)^n / n!
                for (int i=starting_order; i<=max_degree; ++i) {
                    *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * x_over_h_to_i[i]/factorial[i];
                }
            }
        });
    }

    KOKKOS_INLINE_FUNCTION
    void evaluatePartialDerivative(const member_type& teamMember, double* delta, double* workspace, const int dimension, const int max_degree, const int partial_direction, const double h, const double x, const double y, const double z, const int starting_order = 0, const double weight_of_original_value = 0.0, const double weight_of_new_value = 1.0) {
        Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
            const double factorial[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};
            if (dimension==3) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                scratch_vector_type z_over_h_to_i(workspace+2*(max_degree+1), max_degree+1);
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
                for (int n = starting_order; n <= max_degree; n++){
                    for (alphaz = 0; alphaz <= n; alphaz++){
                        s = n - alphaz;
                        for (alphay = 0; alphay <= s; alphay++){
                            alphax = s - alphay;

                            int var_pow[3] = {alphax, alphay, alphaz};
                            var_pow[partial_direction]--;

                            if (var_pow[0]<0 || var_pow[1]<0 || var_pow[2]<0) {
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                            } else {
                                alphaf = factorial[var_pow[0]]*factorial[var_pow[1]]*factorial[var_pow[2]];
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]*z_over_h_to_i[var_pow[2]]/alphaf;
                            }
                            i++;
                        }
                    }
                }
            } else if (dimension==2) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                x_over_h_to_i[0] = 1;
                y_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(x/h);
                    y_over_h_to_i[i] = y_over_h_to_i[i-1]*(y/h);
                }
                // (in 3D) \sum_{p=0}^{p=P} \sum_{k1+k2+k3=n} (x/h)^k1*(y/h)^k2*(z/h)^k3 / (k1!k2!k3!)
                int alphax, alphay;
                double alphaf;
                int i=0;
                for (int n = starting_order; n <= max_degree; n++){
                    for (alphay = 0; alphay <= n; alphay++){
                        alphax = n - alphay;

                        int var_pow[2] = {alphax, alphay};
                        var_pow[partial_direction]--;

                        if (var_pow[0]<0 || var_pow[1]<0) {
                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                        } else {
                            alphaf = factorial[var_pow[0]]*factorial[var_pow[1]];
                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]/alphaf;
                        }
                        i++;
                    }
                }
            } else {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                x_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(x/h);
                }
                int alphax;
                double alphaf;
                int i=0;
                for (int n = starting_order; n <= max_degree; n++){
                    alphax = n;
                
                    int var_pow[1] = {alphax};
                    var_pow[partial_direction]--;

                    if (var_pow[0]<0) {
                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                    } else {
                        alphaf = factorial[var_pow[0]];
                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h * x_over_h_to_i[var_pow[0]]/alphaf;
                    }
                    i++;
                }
            }
        });
    }

    KOKKOS_INLINE_FUNCTION
    void evaluateSecondPartialDerivative(const member_type& teamMember, double* delta, double* workspace, const int dimension, const int max_degree, const int partial_direction_1, const int partial_direction_2, const double h, const double x, const double y, const double z, const int starting_order = 0, const double weight_of_original_value = 0.0, const double weight_of_new_value = 1.0) {
        Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
            const double factorial[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};
            if (dimension==3) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                scratch_vector_type z_over_h_to_i(workspace+2*(max_degree+1), max_degree+1);
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
                for (int n = starting_order; n <= max_degree; n++){
                    for (alphaz = 0; alphaz <= n; alphaz++){
                        s = n - alphaz;
                        for (alphay = 0; alphay <= s; alphay++){
                            alphax = s - alphay;

                            int var_pow[3] = {alphax, alphay, alphaz};
                            var_pow[partial_direction_1]--;
                            var_pow[partial_direction_2]--;

                            if (var_pow[0]<0 || var_pow[1]<0 || var_pow[2]<0) {
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                            } else {
                                alphaf = factorial[var_pow[0]]*factorial[var_pow[1]]*factorial[var_pow[2]];
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]*z_over_h_to_i[var_pow[2]]/alphaf;
                            }
                            i++;
                        }
                    }
                }
            } else if (dimension==2) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                x_over_h_to_i[0] = 1;
                y_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(x/h);
                    y_over_h_to_i[i] = y_over_h_to_i[i-1]*(y/h);
                }
                // (in 3D) \sum_{p=0}^{p=P} \sum_{k1+k2+k3=n} (x/h)^k1*(y/h)^k2*(z/h)^k3 / (k1!k2!k3!)
                int alphax, alphay;
                double alphaf;
                int i=0;
                for (int n = starting_order; n <= max_degree; n++){
                    for (alphay = 0; alphay <= n; alphay++){
                        alphax = n - alphay;

                        int var_pow[2] = {alphax, alphay};
                        var_pow[partial_direction_1]--;
                        var_pow[partial_direction_2]--;

                        if (var_pow[0]<0 || var_pow[1]<0) {
                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                        } else {
                            alphaf = factorial[var_pow[0]]*factorial[var_pow[1]];
                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]/alphaf;
                        }
                        i++;
                    }
                }
            } else {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                x_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(x/h);
                }
                int alphax;
                double alphaf;
                int i=0;
                for (int n = starting_order; n <= max_degree; n++){
                    alphax = n;
                
                    int var_pow[1] = {alphax};
                    var_pow[partial_direction_1]--;
                    var_pow[partial_direction_2]--;

                    if (var_pow[0]<0) {
                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                    } else {
                        alphaf = factorial[var_pow[0]];
                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h * x_over_h_to_i[var_pow[0]]/alphaf;
                    }
                    i++;
                }
            }
        });
    }

} // ScalarTaylorPolynomialBasis

} // Compadre

#endif
