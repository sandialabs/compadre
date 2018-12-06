// @HEADER
// ************************************************************************
//
//                           Compadre Package
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
// IN NO EVENT SHALL SANDIA CORPORATION OR THE CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Paul Kuberry  (pakuber@sandia.gov)
//                    Peter Bosler  (pabosle@sandia.gov), or
//                    Nat Trask     (natrask@sandia.gov)
//
// ************************************************************************
// @HEADER
#ifndef _GMLS_TUTORIAL_HPP_
#define _GMLS_TUTORIAL_HPP_

#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
double device_max(double d1, double d2) {
    return (d1 > d2) ? d1 : d2;
}

KOKKOS_INLINE_FUNCTION
double trueSolution(double x, double y, double z, int order, int dimension) {
    double ans = 0;
    for (int i=0; i<order+1; i++) {
        for (int j=0; j<order+1; j++) {
            for (int k=0; k<order+1; k++) {
                if (i+j+k <= order) {
                    ans += std::pow(x,i)*std::pow(y,j)*std::pow(z,k);
                }
            }
        }
    }
    return ans;
}

KOKKOS_INLINE_FUNCTION
double trueLaplacian(double x, double y, double z, int order, int dimension) {
    double ans = 0;
    for (int i=0; i<order+1; i++) {
        for (int j=0; j<order+1; j++) {
            for (int k=0; k<order+1; k++) {
                if (i+j+k <= order) {
                    ans += device_max(0,i-1)*device_max(0,i)*
                            std::pow(x,device_max(0,i-2))*std::pow(y,j)*std::pow(z,k);
                }
            }
        }
    }
    if (dimension>1) {
        for (int i=0; i<order+1; i++) {
            for (int j=0; j<order+1; j++) {
                for (int k=0; k<order+1; k++) {
                    if (i+j+k <= order) {
                        ans += device_max(0,j-1)*device_max(0,j)*
                                std::pow(x,i)*std::pow(y,device_max(0,j-2))*std::pow(z,k);
                    }
                }
            }
        }
    }
    if (dimension>2) {
        for (int i=0; i<order+1; i++) {
            for (int j=0; j<order+1; j++) {
                for (int k=0; k<order+1; k++) {
                    if (i+j+k <= order) {
                        ans += device_max(0,k-1)*device_max(0,k)*
                                std::pow(x,i)*std::pow(y,j)*std::pow(z,device_max(0,k-2));
                    }
                }
            }
        }
    }
    return ans;
}

KOKKOS_INLINE_FUNCTION
void trueGradient(double* ans, double x, double y, double z, int order, int dimension) {

    for (int i=0; i<order+1; i++) {
        for (int j=0; j<order+1; j++) {
            for (int k=0; k<order+1; k++) {
                if (i+j+k <= order) {
                    ans[0] += device_max(0,i)*
                               std::pow(x,device_max(0,i-1))*std::pow(y,j)*std::pow(z,k);
                }
            }
        }
    }
    if (dimension>1) {
        for (int i=0; i<order+1; i++) {
            for (int j=0; j<order+1; j++) {
                for (int k=0; k<order+1; k++) {
                    if (i+j+k <= order) {
                        ans[1] += device_max(0,j)*
                                   std::pow(x,i)*std::pow(y,device_max(0,j-1))*std::pow(z,k);
                    }
                }
            }
        }
    }
    if (dimension>2) {
        for (int i=0; i<order+1; i++) {
            for (int j=0; j<order+1; j++) {
                for (int k=0; k<order+1; k++) {
                    if (i+j+k <= order) {
                        ans[2] += device_max(0,k)*
                                   std::pow(x,i)*std::pow(y,j)*std::pow(z,device_max(0,k-1));
                    }
                }
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION
double trueDivergence(double x, double y, double z, int order, int dimension) {
    double ans = 0;
    for (int i=0; i<order+1; i++) {
        for (int j=0; j<order+1; j++) {
            for (int k=0; k<order+1; k++) {
                if (i+j+k <= order) {
                    ans += device_max(0,i)*
                            std::pow(x,device_max(0,i-1))*std::pow(y,j)*std::pow(z,k);
                }
            }
        }
    }
    if (dimension>1) {
        for (int i=0; i<order+1; i++) {
            for (int j=0; j<order+1; j++) {
                for (int k=0; k<order+1; k++) {
                    if (i+j+k <= order) {
                        ans += device_max(0,j)*
                                std::pow(x,i)*std::pow(y,device_max(0,j-1))*std::pow(z,k);
                    }
                }
            }
        }
    }
    if (dimension>2) {
        for (int i=0; i<order+1; i++) {
            for (int j=0; j<order+1; j++) {
                for (int k=0; k<order+1; k++) {
                    if (i+j+k <= order) {
                        ans += device_max(0,k)*
                                std::pow(x,i)*std::pow(y,j)*std::pow(z,device_max(0,k-1));
                    }
                }
            }
        }
    }
    return ans;
}

KOKKOS_INLINE_FUNCTION
double divergenceTestSamples(double x, double y, double z, int component, int dimension) {
    // solution can be captured exactly by at least 2rd order
    switch (component) {
    case 0:
        return x*x + y*y - z*z;
    case 1:
        return 2*x +3*y + 4*z;
    default:
        return -21*x*y + 3*z*x*y + 4*z;
    }
}

KOKKOS_INLINE_FUNCTION
double divergenceTestSolution(double x, double y, double z, int dimension) {
    switch (dimension) {
    case 1:
        // returns divergence of divergenceTestSamples
        return 2*x;
    case 2:
        return 2*x + 3;
    default:
        return 2*x + 3 + 3*x*y + 4;
    }
}

KOKKOS_INLINE_FUNCTION
double curlTestSolution(double x, double y, double z, int component, int dimension) {
    if (dimension==3) {
        // returns curl of divergenceTestSamples
        switch (component) {
        case 0:
            // u3,y- u2,z
            return (-21*x + 3*z*x) - 4;
        case 1:
            // -u3,x + u1,z
            return (21*y - 3*z*y) - 2*z;
        default:
            // u2,x - u1,y
            return 2 - 2*y;
        }
    } else if (dimension==2) {
        switch (component) {
        case 0:
            // u2,y
            return 3;
        default:
            // -u1,x
            return -2*x;
        }
    } else {
        return 0;
    }
}


/** Standard GMLS Example 
 *
 *  Exercises GMLS operator evaluation with data over various orders and numbers of targets for targets including point evaluation, Laplacian, divergence, curl, and gradient.
 */
int main (int argc, char* args[]);

/**
 * \example "GMLS Tutorial" based on GMLS_Device.cpp
 * \section ex GMLS Example with Device Views
 *
 * This tutorial sets up a batch of GMLS problems, solves the minimization problems, and applies the coefficients produced to data.
 * 
 * \section ex1a Parse Command Line Arguments
 * \snippet GMLS_Device.cpp Parse Command Line Arguments
 *
 * \section ex1b Setting Up The Point Cloud
 * \snippet GMLS_Device.cpp Setting Up The Point Cloud
 *
 * \section ex1c Performing Neighbor Search
 * \snippet GMLS_Device.cpp Performing Neighbor Search
 *
 * \section ex2 Creating The Data
 * \snippet GMLS_Device.cpp Creating The Data
 *
 * \section ex3 Setting Up The GMLS Object
 * \snippet GMLS_Device.cpp Setting Up The GMLS Object
 * 
 * \section ex4 Apply GMLS Alphas To Data
 * \snippet GMLS_Device.cpp Apply GMLS Alphas To Data
 *
 * \section ex5 Check That Solutions Are Correct
 * \snippet GMLS_Device.cpp Check That Solutions Are Correct
 *
 * \section ex6 Finalize Program
 * \snippet GMLS_Device.cpp Finalize Program
 */ 

#endif
