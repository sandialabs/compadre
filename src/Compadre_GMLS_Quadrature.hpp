#ifndef _COMPADRE_GMLS_QUADRATURE_HPP_
#define _COMPADRE_GMLS_QUADRATURE_HPP_

#include "Compadre_GMLS.hpp"

namespace Compadre {

void GMLS::generate1DQuadrature() {

    _quadrature_weights = Kokkos::View<double*, layout_type>("1d quadrature weights", _number_of_quadrature_points);
    _parameterized_quadrature_sites = Kokkos::View<double*, layout_type>("1d quadrature sites", _number_of_quadrature_points);

    Kokkos::View<double*, layout_type>::HostMirror quadrature_weights = create_mirror(_quadrature_weights);
    Kokkos::View<double*, layout_type>::HostMirror parameterized_quadrature_sites = create_mirror(_parameterized_quadrature_sites);

    switch (_number_of_quadrature_points) {
    case 1:
        quadrature_weights[0] = 1.0;
        parameterized_quadrature_sites[0] = 0.5;
        break;
    case 2:
        quadrature_weights[0] = 0.5;
        quadrature_weights[1] = 0.5;
        parameterized_quadrature_sites[0] = 0.5*(1-std::sqrt(1./3.));
        parameterized_quadrature_sites[1] = 0.5*(1+std::sqrt(1./3.));
        break;
    case 3:
        quadrature_weights[0] = 2.5/9.;
        quadrature_weights[1] = 2.5/9;
        quadrature_weights[2] = 4./9.;
        parameterized_quadrature_sites[0] = 0.5*(1-std::sqrt(3./5.));
        parameterized_quadrature_sites[1] = 0.5*(1+std::sqrt(3./5.));
        parameterized_quadrature_sites[2] = 0.5;
        break;
    case 4:
        quadrature_weights[0] = (18+std::sqrt(30))/72.;
        quadrature_weights[1] = (18+std::sqrt(30))/72.;
        quadrature_weights[2] = (18-std::sqrt(30))/72.;
        quadrature_weights[3] = (18-std::sqrt(30))/72.;
        parameterized_quadrature_sites[0] = 0.5*(1-std::sqrt(3./7.-2./7.*std::sqrt(6./5.)));
        parameterized_quadrature_sites[1] = 0.5*(1+std::sqrt(3./7.-2./7.*std::sqrt(6./5.)));
        parameterized_quadrature_sites[2] = 0.5*(1-std::sqrt(3./7.+2./7.*std::sqrt(6./5.)));
        parameterized_quadrature_sites[3] = 0.5*(1+std::sqrt(3./7.+2./7.*std::sqrt(6./5.)));
        break;
    case 5:
        quadrature_weights[0] = 128./450.;
        quadrature_weights[1] = (322+13*std::sqrt(70))/1800.;
        quadrature_weights[2] = (322+13*std::sqrt(70))/1800.;
        quadrature_weights[3] = (322-13*std::sqrt(70))/1800.;
        quadrature_weights[4] = (322-13*std::sqrt(70))/1800.;
        parameterized_quadrature_sites[0] = 0.5;
        parameterized_quadrature_sites[1] = 0.5*(1+1./3.*std::sqrt(5-2*std::sqrt(10./7.)));
        parameterized_quadrature_sites[2] = 0.5*(1-1./3.*std::sqrt(5-2*std::sqrt(10./7.)));
        parameterized_quadrature_sites[3] = 0.5*(1+1./3.*std::sqrt(5+2*std::sqrt(10./7.)));
        parameterized_quadrature_sites[4] = 0.5*(1-1./3.*std::sqrt(5+2*std::sqrt(10./7.)));
        break;
    default:
        compadre_assert_release(false && "Unsupported number of quadrature points.");
        quadrature_weights[0] = 0.5;
        quadrature_weights[1] = 0.5;
        parameterized_quadrature_sites[0] = 0.5*(1-std::sqrt(1./3.));
        parameterized_quadrature_sites[1] = 0.5*(1+std::sqrt(1./3.));
        for (int i=2; i<_number_of_quadrature_points; ++i) {
            quadrature_weights[i] = 0;
            parameterized_quadrature_sites[i] = 0;
        }
    }

    Kokkos::deep_copy(_quadrature_weights, quadrature_weights);
    Kokkos::deep_copy(_parameterized_quadrature_sites, parameterized_quadrature_sites);
}

}; // Compadre

#endif
