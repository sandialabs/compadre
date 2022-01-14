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
        return std::pow(ScalarTaylorPolynomialBasis::getSize(degree, 1), dimension);
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

    /*! \brief Evaluates the hard-coded divergence-free polynomial basis up to order 4 for 3D
     *
        Polynomial degree is inferred from np
        \warning This function is deprecated.
        \param np                        [in] - basis component number
        \param sx                        [in] - x coordinate (already shifted by target and scaled)
        \param sy                        [in] - y coordinate (already shifted by target and scaled)
        \param sz                        [in] - z coordinate (already shifted by target and scaled)
    */
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

            case 65: return XYZ(sx*sx*sx*sx         ,-4.0*sx*sx*sx*sy    ,                   0);
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

    /*! \brief Evaluates the hard-coded divergence-free polynomial basis up to order 4 for 2D
     *
        Polynomial degree is inferred from np
        \warning This function is deprecated.
        \param np                        [in] - basis component number
        \param sx                        [in] - x coordinate (already shifted by target and scaled)
        \param sy                        [in] - y coordinate (already shifted by target and scaled)
    */
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

    /*! \brief Evaluates the hard-coded first partial derivative of the divergence-free polynomial basis up to order 4 for 3D
     *
        Polynomial degree is inferred from np
        \warning This function is deprecated.
        \param np                        [in] - basis component number
        \param partial_direction         [in] - partial derivative direction
        \param sx                        [in] - x coordinate (already shifted by target and scaled)
        \param sy                        [in] - y coordinate (already shifted by target and scaled)
        \param sz                        [in] - z coordinate (already shifted by target and scaled)
    */
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

                case 65: return XYZ( 4*sx*sx*sx/h       ,-12.0*sx*sx*sy/h    ,                   0);
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

                case 65: return XYZ(                   0,     -4.0*sx*sx*sx/h,                   0);
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

    /*! \brief Evaluates the hard-coded first partial derivative of the divergence-free polynomial basis up to order 4 for 2D
     *
        Polynomial degree is inferred from np
        \warning This function is deprecated.
        \param np                        [in] - basis component number
        \param partial_direction         [in] - partial derivative direction
        \param sx                        [in] - x coordinate (already shifted by target and scaled)
        \param sy                        [in] - y coordinate (already shifted by target and scaled)
    */
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
} // BernsteinPolynomialBasis
} // Compadre

#endif
