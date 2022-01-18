#ifndef _COMPADRE_GMLS_BERNSTEIN_BASIS_HPP_
#define _COMPADRE_GMLS_BERNSTEIN_BASIS_HPP_

#include "Compadre_GMLS.hpp"
#include "Compadre_ScalarTaylorPolynomial.hpp"

namespace Compadre {
/*!  \brief Definition of scalar Bernstein polynomial basis
   
   For order P, the sum of the basis is defined by:
   \li in 1D) \f[\sum_{n=0}^{n=P} \frac{P!}{n!(P-n)!}*\left(\frac{x}{2h}+\frac{1}{2}\right)^n*\left(1-\frac{x}{2h}-\frac{1}{2}\right)^{P-n}\f]
   \li in 2D) \f[\sum_{n_1=0}^{n_1=P}\sum_{n_2=0}^{n_2=P} \frac{P!}{n_1!(P-n_1)!}*\frac{P!}{n_2!(P-n_2)!}*\left(\frac{x}{2h}+\frac{1}{2}\right)^{n_1}*\left(1-\frac{x}{2h}-\frac{1}{2}\right)^{P-n_1}*\left(\frac{y}{2h}+\frac{1}{2}\right)^{n_2}*\left(1-\frac{y}{2h}-\frac{1}{2}\right)^{P-n_2}\f]
   \li in 3D) \f[\sum_{n_1=0}^{n_1=P}\sum_{n_2=0}^{n_2=P}\sum_{n_3=0}^{n_3=P} \frac{P!}{n_1!(P-n_1)!}*\frac{P!}{n_2!(P-n_2)!}*\frac{P!}{n_3!(P-n_3)!}*\left(\frac{x}{2h}+\frac{1}{2}\right)^{n_1}*\left(1-\frac{x}{2h}-\frac{1}{2}\right)^{P-n_1}*\f]\f[\left(\frac{y}{2h}+\frac{1}{2}\right)^{n_2}*\left(1-\frac{y}{2h}-\frac{1}{2}\right)^{P-n_2}*\left(\frac{z}{2h}+\frac{1}{2}\right)^{n_3}*\left(1-\frac{z}{2h}-\frac{1}{2}\right)^{P-n_3}\f]
*/
namespace BernsteinPolynomialBasis {

    /*! \brief Returns size of basis
        \param degree               [in] - highest degree of polynomial
        \param dimension            [in] - spatial dimension to evaluate
    */
    KOKKOS_INLINE_FUNCTION
    int getSize(const int degree, const int dimension) {
        return pown(ScalarTaylorPolynomialBasis::getSize(degree, int(1)), dimension);
    }

    /*! \brief Evaluates the Bernstein polynomial basis
     *  delta[j] = weight_of_original_value * delta[j] + weight_of_new_value * (calculation of this function)
        \param delta                [in/out] - scratch space that is allocated so that each thread has its own copy. Must be at least as large as the _basis_multipler*the dimension of the polynomial basis.
        \param workspace            [in/out] - scratch space that is allocated so that each thread has its own copy. Must be at least as large as the _poly_order*the spatial dimension of the polynomial basis.
        \param dimension                [in] - spatial dimension to evaluate
        \param max_degree               [in] - highest degree of polynomial
        \param h                        [in] - epsilon/window size
        \param x                        [in] - x coordinate (already shifted by target)
        \param y                        [in] - y coordinate (already shifted by target)
        \param z                        [in] - z coordinate (already shifted by target)
        \param starting_order           [in] - polynomial order to start with (default=0)
        \param weight_of_original_value [in] - weighting to assign to original value in delta (default=0, replace, 1-increment current value)
        \param weight_of_new_value      [in] - weighting to assign to new contribution        (default=1, add to/replace)
    */
    KOKKOS_INLINE_FUNCTION
    void evaluate(const member_type& teamMember, double* delta, double* workspace, const int dimension, const int max_degree, const int component, const double h, const double x, const double y, const double z, const int starting_order = 0, const double weight_of_original_value = 0.0, const double weight_of_new_value = 1.0) {
        Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
            const double factorial[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};
            if (dimension==3) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                scratch_vector_type z_over_h_to_i(workspace+2*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_x_over_h_to_i(workspace+3*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_y_over_h_to_i(workspace+4*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_z_over_h_to_i(workspace+5*(max_degree+1), max_degree+1);
                x_over_h_to_i[0] = 1;
                y_over_h_to_i[0] = 1;
                z_over_h_to_i[0] = 1;
                one_minus_x_over_h_to_i[0] = 1;
                one_minus_y_over_h_to_i[0] = 1;
                one_minus_z_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(0.5*x/h+0.5);
                    y_over_h_to_i[i] = y_over_h_to_i[i-1]*(0.5*y/h+0.5);
                    z_over_h_to_i[i] = z_over_h_to_i[i-1]*(0.5*z/h+0.5);
                    one_minus_x_over_h_to_i[i] = one_minus_x_over_h_to_i[i-1]*(1.0-(0.5*x/h+0.5));
                    one_minus_y_over_h_to_i[i] = one_minus_y_over_h_to_i[i-1]*(1.0-(0.5*y/h+0.5));
                    one_minus_z_over_h_to_i[i] = one_minus_z_over_h_to_i[i-1]*(1.0-(0.5*z/h+0.5));
                }
                // (in 3D) \sum_{n_1=0}^{n_1=P}\sum_{n_2=0}^{n_2=P}\sum_{n_3=0}^{n_3=P} 
                // \frac{P!}{n_1!(P-n_1)!}*\frac{P!}{n_2!(P-n_2)!}*\frac{P!}{n_3!(P-n_3)!}*
                // \left(\frac{x}{2h}+\frac{1}{2}\right)^{n_1}*\left(1-\frac{x}{2h}-\frac{1}{2}\right)^{P-n_1}*
                // \left(\frac{y}{2h}+\frac{1}{2}\right)^{n_2}*\left(1-\frac{y}{2h}-\frac{1}{2}\right)^{P-n_2}*
                // \left(\frac{z}{2h}+\frac{1}{2}\right)^{n_3}*\left(1-\frac{z}{2h}-\frac{1}{2}\right)^{P-n_3}
                int alphax, alphay, alphaz;
                double alphaf;
                int i=0;
                for (int n1 = starting_order; n1 <= max_degree; n1++){
                    int degree_choose_n1 = factorial[max_degree]/(factorial[n1]*factorial[max_degree-n1]);
                    for (int n2 = starting_order; n2 <= max_degree; n2++){
                        int degree_choose_n2 = factorial[max_degree]/(factorial[n2]*factorial[max_degree-n2]);
                        for (int n3 = starting_order; n3 <= max_degree; n3++){
                            int degree_choose_n3 = factorial[max_degree]/(factorial[n3]*factorial[max_degree-n3]);
                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * degree_choose_n1*degree_choose_n2*degree_choose_n3*x_over_h_to_i[n1]*one_minus_x_over_h_to_i[max_degree-n1]*y_over_h_to_i[n2]*one_minus_y_over_h_to_i[max_degree-n2]*z_over_h_to_i[n3]*one_minus_z_over_h_to_i[max_degree-n3];
                            i++;
                        }
                    }
                }
            } else if (dimension==2) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_x_over_h_to_i(workspace+2*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_y_over_h_to_i(workspace+3*(max_degree+1), max_degree+1);
                x_over_h_to_i[0] = 1;
                y_over_h_to_i[0] = 1;
                one_minus_x_over_h_to_i[0] = 1;
                one_minus_y_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(0.5*x/h+0.5);
                    y_over_h_to_i[i] = y_over_h_to_i[i-1]*(0.5*y/h+0.5);
                    one_minus_x_over_h_to_i[i] = one_minus_x_over_h_to_i[i-1]*(1.0-(0.5*x/h+0.5));
                    one_minus_y_over_h_to_i[i] = one_minus_y_over_h_to_i[i-1]*(1.0-(0.5*y/h+0.5));
                }
                // (in 2D) \sum_{n_1=0}^{n_1=P}\sum_{n_2=0}^{n_2=P} \frac{P!}{n_1!(P-n_1)!}*
                // \frac{P!}{n_2!(P-n_2)!}*\left(\frac{x}{2h}+\frac{1}{2}\right)^{n_1}*
                // \left(1-\frac{x}{2h}-\frac{1}{2}\right)^{P-n_1}*\left(\frac{y}{2h}+\frac{1}{2}\right)^{n_2}*
                // \left(1-\frac{y}{2h}-\frac{1}{2}\right)^{P-n_2}
                int alphax, alphay;
                double alphaf;
                int i = 0;
                for (int n1 = starting_order; n1 <= max_degree; n1++){
                    int degree_choose_n1 = factorial[max_degree]/(factorial[n1]*factorial[max_degree-n1]);
                    for (int n2 = starting_order; n2 <= max_degree; n2++){
                        int degree_choose_n2 = factorial[max_degree]/(factorial[n2]*factorial[max_degree-n2]);
                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * degree_choose_n1*degree_choose_n2*x_over_h_to_i[n1]*one_minus_x_over_h_to_i[max_degree-n1]*y_over_h_to_i[n2]*one_minus_y_over_h_to_i[max_degree-n2];
                        i++;
                    }
                }
            } else {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type one_minus_x_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                x_over_h_to_i[0] = 1;
                one_minus_x_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(0.5*x/h+0.5);
                    one_minus_x_over_h_to_i[i] = one_minus_x_over_h_to_i[i-1]*(1.0-(0.5*x/h+0.5));
                }
                // (in 1D) \sum_{n=0}^{n=P} \frac{P!}{n!(P-n)!}*\left(\frac{x}{2h}+\frac{1}{2}\right)^n*
                // \left(1-\frac{x}{2h}-\frac{1}{2}\right)^{P-n}
                int i = 0;
                for (int n=starting_order; n<=max_degree; ++n) {
                    int degree_choose_n = factorial[max_degree]/(factorial[n]*factorial[max_degree-n]);
                    *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * degree_choose_n*x_over_h_to_i[n]*one_minus_x_over_h_to_i[max_degree-n];
                    i++;
                }
            }
        });
    }

    /*! \brief Evaluates the first partial derivatives of the divergence-free polynomial basis
     *  delta[j] = weight_of_original_value * delta[j] + weight_of_new_value * (calculation of this function)
        \param delta                [in/out] - scratch space that is allocated so that each thread has its own copy. Must be at least as large as the _basis_multipler*the dimension of the polynomial basis.
        \param workspace            [in/out] - scratch space that is allocated so that each thread has its own copy. Must be at least as large as the _poly_order*the spatial dimension of the polynomial basis.
        \param dimension                [in] - spatial dimension to evaluate
        \param max_degree               [in] - highest degree of polynomial
        \param partial_direction        [in] - direction in which to take partial derivative
        \param h                        [in] - epsilon/window size
        \param x                        [in] - x coordinate (already shifted by target)
        \param y                        [in] - y coordinate (already shifted by target)
        \param z                        [in] - z coordinate (already shifted by target)
        \param starting_order           [in] - polynomial order to start with (default=0)
        \param weight_of_original_value [in] - weighting to assign to original value in delta (default=0, replace, 1-increment current value)
        \param weight_of_new_value      [in] - weighting to assign to new contribution        (default=1, add to/replace)
    */
    KOKKOS_INLINE_FUNCTION
    void evaluatePartialDerivative(const member_type& teamMember, double* delta, double* workspace, const int dimension, const int max_degree, const int component, const int partial_direction, const double h, const double x, const double y, const double z, const int starting_order = 0, const double weight_of_original_value = 0.0, const double weight_of_new_value = 1.0) {
        Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
            const double factorial[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};
            if (dimension==3) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                scratch_vector_type z_over_h_to_i(workspace+2*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_x_over_h_to_i(workspace+3*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_y_over_h_to_i(workspace+4*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_z_over_h_to_i(workspace+5*(max_degree+1), max_degree+1);
                x_over_h_to_i[0] = 1;
                y_over_h_to_i[0] = 1;
                z_over_h_to_i[0] = 1;
                one_minus_x_over_h_to_i[0] = 1;
                one_minus_y_over_h_to_i[0] = 1;
                one_minus_z_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(0.5*x/h+0.5);
                    y_over_h_to_i[i] = y_over_h_to_i[i-1]*(0.5*y/h+0.5);
                    z_over_h_to_i[i] = z_over_h_to_i[i-1]*(0.5*z/h+0.5);
                    one_minus_x_over_h_to_i[i] = one_minus_x_over_h_to_i[i-1]*(1.0-(0.5*x/h+0.5));
                    one_minus_y_over_h_to_i[i] = one_minus_y_over_h_to_i[i-1]*(1.0-(0.5*y/h+0.5));
                    one_minus_z_over_h_to_i[i] = one_minus_z_over_h_to_i[i-1]*(1.0-(0.5*z/h+0.5));
                }
                int i=0;
                for (int d=0; d<dimension; ++d) {
                    if ((d+1)==dimension) {
                        if (component==d) {
                            // use 2D partial derivative of scalar basis definition
                            // (in 2D) \sum_{n=0}^{n=P} \sum_{k=0}^{k=n} (x/h)^(n-k)*(y/h)^k / ((n-k)!k!)
                            int alphax, alphay;
                            double alphaf;
                            for (int n = starting_order; n <= max_degree; n++){
                                for (alphay = 0; alphay <= n; alphay++){
                                    alphax = n - alphay;

                                    int var_pow[3] = {alphax, alphay, 0};
                                    var_pow[partial_direction]--;
                
                                    if (var_pow[0]<0 || var_pow[1]<0 || var_pow[2]<0) {
                                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                    } else {
                                        alphaf = factorial[var_pow[0]]*factorial[var_pow[1]];
                                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]/alphaf;
                                    }
                                    i++;
                                }
                            }
                        } else {
                            for (int j=0; j<ScalarTaylorPolynomialBasis::getSize(max_degree, dimension-1); ++j) { 
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0;
                                i++;
                            }
                        }
                    } else {
                        if (component==d) {
                            // use 3D partial derivative of scalar basis definition
                            // (in 3D) \sum_{p=0}^{p=P} \sum_{k1+k2+k3=n} (x/h)^k1*(y/h)^k2*(z/h)^k3 / (k1!k2!k3!)
                            int alphax, alphay, alphaz;
                            double alphaf;
                            int s=0;
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
                        } else if (component==(d+1)%3) {
                            for (int j=0; j<ScalarTaylorPolynomialBasis::getSize(max_degree, dimension); ++j) { 
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0;
                                i++;
                            }
                        } else {
                            // (in 3D)
                            int alphax, alphay, alphaz;
                            double alphaf;
                            int s=0;
                            for (int n = starting_order; n <= max_degree; n++){
                                for (alphaz = 0; alphaz <= n; alphaz++){
                                    s = n - alphaz;
                                    for (alphay = 0; alphay <= s; alphay++){
                                        alphax = s - alphay;

                                        int var_pow[3] = {alphax, alphay, alphaz};
                                        var_pow[d]--;

                                        if (var_pow[0]<0 || var_pow[1]<0 || var_pow[2]<0) {
                                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                        } else {
                                            var_pow[component]++;
                                            var_pow[partial_direction]--;
                                            if (var_pow[0]<0 || var_pow[1]<0 || var_pow[2]<0) {
                                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                            } else {
                                                alphaf = factorial[var_pow[0]]*factorial[var_pow[1]]*factorial[var_pow[2]];
                                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * -1.0/h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]*z_over_h_to_i[var_pow[2]]/alphaf;
                                            }
                                        }
                                        i++;
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (dimension==2) {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type y_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_x_over_h_to_i(workspace+2*(max_degree+1), max_degree+1);
                scratch_vector_type one_minus_y_over_h_to_i(workspace+3*(max_degree+1), max_degree+1);
                x_over_h_to_i[0] = 1;
                y_over_h_to_i[0] = 1;
                one_minus_x_over_h_to_i[0] = 1;
                one_minus_y_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(0.5*x/h+0.5);
                    y_over_h_to_i[i] = y_over_h_to_i[i-1]*(0.5*y/h+0.5);
                    one_minus_x_over_h_to_i[i] = one_minus_x_over_h_to_i[i-1]*(1.0-(0.5*x/h+0.5));
                    one_minus_y_over_h_to_i[i] = one_minus_y_over_h_to_i[i-1]*(1.0-(0.5*y/h+0.5));
                }
                int i=0;
                for (int d=0; d<dimension; ++d) {
                    if ((d+1)==dimension) {
                        if (component==d) {
                            // use 1D partial derivative of scalar basis definition
                            int alphax;
                            double alphaf;
                            for (int n = starting_order; n <= max_degree; n++){
                                alphax = n;
                            
                                int var_pow[2] = {alphax, 0};
                                var_pow[partial_direction]--;

                                if (var_pow[0]<0 || var_pow[1]<0) {
                                    *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                } else {
                                    alphaf = factorial[var_pow[0]];
                                    *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h * x_over_h_to_i[var_pow[0]]/alphaf;
                                }
                                i++;
                            }
                        } else {
                            for (int j=0; j<ScalarTaylorPolynomialBasis::getSize(max_degree, dimension-1); ++j) { 
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0;
                                i++;
                            }
                        }
                    } else {
                        if (component==d) {
                            // use 2D partial derivative of scalar basis definition
                            int alphax, alphay;
                            double alphaf;
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
                            // (in 2D)
                            int alphax, alphay;
                            double alphaf;
                            for (int n = starting_order; n <= max_degree; n++){
                                for (alphay = 0; alphay <= n; alphay++){
                                    alphax = n - alphay;

                                    int var_pow[2] = {alphax, alphay};
                                    var_pow[d]--;

                                    if (var_pow[0]<0 || var_pow[1]<0) {
                                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                    } else {
                                        var_pow[component]++;
                                        var_pow[partial_direction]--;
                                        if (var_pow[0]<0 || var_pow[1]<0) {
                                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                        } else {
                                            alphaf = factorial[var_pow[0]]*factorial[var_pow[1]];
                                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * -1.0/h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]/alphaf;
                                        }
                                    }
                                    i++;
                                }
                            }
                        }
                    }
                }

            } else {
                scratch_vector_type x_over_h_to_i(workspace, max_degree+1);
                scratch_vector_type one_minus_x_over_h_to_i(workspace+1*(max_degree+1), max_degree+1);
                x_over_h_to_i[0] = 1;
                one_minus_x_over_h_to_i[0] = 1;
                for (int i=1; i<=max_degree; ++i) {
                    x_over_h_to_i[i] = x_over_h_to_i[i-1]*(0.5*x/h+0.5);
                    one_minus_x_over_h_to_i[i] = one_minus_x_over_h_to_i[i-1]*(1.0-(0.5*x/h+0.5));
                }
                compadre_kernel_assert_release((false) && "Divergence-free basis only defined or dimensions 2 and 3.");
            }
        });
    }

    /*! \brief Evaluates the second partial derivatives of the divergence-free polynomial basis
     *  delta[j] = weight_of_original_value * delta[j] + weight_of_new_value * (calculation of this function)
        \param delta                [in/out] - scratch space that is allocated so that each thread has its own copy. Must be at least as large as the _basis_multipler*the dimension of the polynomial basis.
        \param workspace            [in/out] - scratch space that is allocated so that each thread has its own copy. Must be at least as large as the _poly_order*the spatial dimension of the polynomial basis.
        \param dimension                [in] - spatial dimension to evaluate
        \param max_degree               [in] - highest degree of polynomial
        \param partial_direction_1      [in] - direction in which to take first partial derivative
        \param partial_direction_2      [in] - direction in which to take second partial derivative
        \param h                        [in] - epsilon/window size
        \param x                        [in] - x coordinate (already shifted by target)
        \param y                        [in] - y coordinate (already shifted by target)
        \param z                        [in] - z coordinate (already shifted by target)
        \param starting_order           [in] - polynomial order to start with (default=0)
        \param weight_of_original_value [in] - weighting to assign to original value in delta (default=0, replace, 1-increment current value)
        \param weight_of_new_value      [in] - weighting to assign to new contribution        (default=1, add to/replace)
    */
    KOKKOS_INLINE_FUNCTION
    void evaluateSecondPartialDerivative(const member_type& teamMember, double* delta, double* workspace, const int dimension, const int max_degree, const int component, const int partial_direction_1, const int partial_direction_2, const double h, const double x, const double y, const double z, const int starting_order = 0, const double weight_of_original_value = 0.0, const double weight_of_new_value = 1.0) {
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
                int i=0;
                for (int d=0; d<dimension; ++d) {
                    if ((d+1)==dimension) {
                        if (component==d) {
                            // use 2D partial derivative of scalar basis definition
                            // (in 2D) \sum_{n=0}^{n=P} \sum_{k=0}^{k=n} (x/h)^(n-k)*(y/h)^k / ((n-k)!k!)
                            int alphax, alphay;
                            double alphaf;
                            for (int n = starting_order; n <= max_degree; n++){
                                for (alphay = 0; alphay <= n; alphay++){
                                    alphax = n - alphay;

                                    int var_pow[3] = {alphax, alphay, 0};
                                    var_pow[partial_direction_1]--;
                                    var_pow[partial_direction_2]--;
                
                                    if (var_pow[0]<0 || var_pow[1]<0 || var_pow[2]<0) {
                                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                    } else {
                                        alphaf = factorial[var_pow[0]]*factorial[var_pow[1]];
                                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h/h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]/alphaf;
                                    }
                                    i++;
                                }
                            }
                        } else {
                            for (int j=0; j<ScalarTaylorPolynomialBasis::getSize(max_degree, dimension-1); ++j) { 
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0;
                                i++;
                            }
                        }
                    } else {
                        if (component==d) {
                            // use 3D partial derivative of scalar basis definition
                            // (in 3D) \sum_{p=0}^{p=P} \sum_{k1+k2+k3=n} (x/h)^k1*(y/h)^k2*(z/h)^k3 / (k1!k2!k3!)
                            int alphax, alphay, alphaz;
                            double alphaf;
                            int s=0;
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
                                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h/h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]*z_over_h_to_i[var_pow[2]]/alphaf;
                                        }
                                        i++;
                                    }
                                }
                            }
                        } else if (component==(d+1)%3) {
                            for (int j=0; j<ScalarTaylorPolynomialBasis::getSize(max_degree, dimension); ++j) { 
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0;
                                i++;
                            }
                        } else {
                            // (in 3D)
                            int alphax, alphay, alphaz;
                            double alphaf;
                            int s=0;
                            for (int n = starting_order; n <= max_degree; n++){
                                for (alphaz = 0; alphaz <= n; alphaz++){
                                    s = n - alphaz;
                                    for (alphay = 0; alphay <= s; alphay++){
                                        alphax = s - alphay;

                                        int var_pow[3] = {alphax, alphay, alphaz};
                                        var_pow[d]--;

                                        if (var_pow[0]<0 || var_pow[1]<0 || var_pow[2]<0) {
                                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                        } else {
                                            var_pow[component]++;
                                            var_pow[partial_direction_1]--;
                                            var_pow[partial_direction_2]--;
                                            if (var_pow[0]<0 || var_pow[1]<0 || var_pow[2]<0) {
                                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                            } else {
                                                alphaf = factorial[var_pow[0]]*factorial[var_pow[1]]*factorial[var_pow[2]];
                                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * -1.0/h/h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]*z_over_h_to_i[var_pow[2]]/alphaf;
                                            }
                                        }
                                        i++;
                                    }
                                }
                            }
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
                int i=0;
                for (int d=0; d<dimension; ++d) {
                    if ((d+1)==dimension) {
                        if (component==d) {
                            // use 1D partial derivative of scalar basis definition
                            int alphax;
                            double alphaf;
                            for (int n = starting_order; n <= max_degree; n++){
                                alphax = n;
                            
                                int var_pow[2] = {alphax, 0};
                                var_pow[partial_direction_1]--;
                                var_pow[partial_direction_2]--;

                                if (var_pow[0]<0 || var_pow[1]<0) {
                                    *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                } else {
                                    alphaf = factorial[var_pow[0]];
                                    *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h/h * x_over_h_to_i[var_pow[0]]/alphaf;
                                }
                                i++;
                            }
                        } else {
                            for (int j=0; j<ScalarTaylorPolynomialBasis::getSize(max_degree, dimension-1); ++j) { 
                                *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0;
                                i++;
                            }
                        }
                    } else {
                        if (component==d) {
                            // use 2D partial derivative of scalar basis definition
                            int alphax, alphay;
                            double alphaf;
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
                                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 1./h/h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]/alphaf;
                                    }
                                    i++;
                                }
                            }
                        } else {
                            // (in 2D)
                            int alphax, alphay;
                            double alphaf;
                            for (int n = starting_order; n <= max_degree; n++){
                                for (alphay = 0; alphay <= n; alphay++){
                                    alphax = n - alphay;

                                    int var_pow[2] = {alphax, alphay};
                                    var_pow[d]--;

                                    if (var_pow[0]<0 || var_pow[1]<0) {
                                        *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                    } else {
                                        var_pow[component]++;
                                        var_pow[partial_direction_1]--;
                                        var_pow[partial_direction_2]--;
                                        if (var_pow[0]<0 || var_pow[1]<0) {
                                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * 0.0;
                                        } else {
                                            alphaf = factorial[var_pow[0]]*factorial[var_pow[1]];
                                            *(delta+i) = weight_of_original_value * *(delta+i) + weight_of_new_value * -1.0/h/h * x_over_h_to_i[var_pow[0]]*y_over_h_to_i[var_pow[1]]/alphaf;
                                        }
                                    }
                                    i++;
                                }
                            }
                        }
                    }
                }

            } else {
                compadre_kernel_assert_release((false) && "Divergence-free basis only defined or dimensions 2 and 3.");
            }
        });
    }

} // BernsteinPolynomialBasis
} // Compadre

#endif
