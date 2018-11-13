#ifndef _COMPADRE_GMLS_TARGETS_HPP_
#define _COMPADRE_GMLS_TARGETS_HPP_

#include "Compadre_GMLS.hpp"

namespace Compadre {

KOKKOS_INLINE_FUNCTION
void GMLS::computeTargetFunctionals(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type P_target_row, const int basis_multiplier_component) const {

    const int target_index = teamMember.league_rank();

    const int target_NP = this->getNP(_poly_order, _dimensions);
    for (int i=0; i<_operations.size(); ++i) {
        if (_operations(i) == TargetOperation::ScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                this->calcPij(P_target_row.data()+_lro_total_offsets[i]*target_NP, target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, SamplingFunctional::PointSample);
            });
        } else if (_operations(i) == TargetOperation::VectorPointEvaluation) {
            ASSERT_WITH_MESSAGE(false, "Functionality for vectors not yet available. Currently performed using ScalarPointEvaluation for each component.");
        } else if (_operations(i) == TargetOperation::LaplacianOfScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                const int offset = _lro_total_offsets[i]*target_NP;
                for (int j=0; j<target_NP; ++j) {
                    P_target_row(offset + j, basis_multiplier_component) = 0;
                }
                switch (_dimensions) {
                case 3:
                    P_target_row(offset + 4, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
                    P_target_row(offset + 6, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
                    P_target_row(offset + 9, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
                    break;
                case 2:
                    P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
                    P_target_row(offset + 5, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
                    break;
                default:
                    P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
                }
            });
        } else if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                int offset = (_lro_total_offsets[i]+0)*target_NP;
                for (int j=0; j<target_NP; ++j) {
                    P_target_row(offset + j, basis_multiplier_component) = 0;
                }
                P_target_row(offset + 1, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//                this->calcGradientPij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, SamplingFunctional::PointSample);

                if (_dimensions>1) {
                    offset = (_lro_total_offsets[i]+1)*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                    }
                    P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//                    this->calcGradientPij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, SamplingFunctional::PointSample);
                }

                if (_dimensions>2) {
                    offset = (_lro_total_offsets[i]+2)*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                    }
                    P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//                    this->calcGradientPij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, SamplingFunctional::PointSample);
                }
            });
        } else if (_operations(i) == TargetOperation::PartialXOfScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                int offset = _lro_total_offsets[i]*target_NP;
                for (int j=0; j<target_NP; ++j) {
                    P_target_row(offset + j, basis_multiplier_component) = 0;
                }
                P_target_row(offset + 1, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//                this->calcGradientPij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, SamplingFunctional::PointSample);
                });
        } else if (_operations(i) == TargetOperation::PartialYOfScalarPointEvaluation) {
            if (_dimensions>1) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = _lro_total_offsets[i]*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                    }
                    P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//                    this->calcGradientPij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, SamplingFunctional::PointSample);
                });
            }
        } else if (_operations(i) == TargetOperation::PartialZOfScalarPointEvaluation) {
            if (_dimensions>2) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = _lro_total_offsets[i]*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                    }
                    P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//                    this->calcGradientPij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, SamplingFunctional::PointSample);
                });
            }
        } else if (_operations(i) == TargetOperation::DivergenceOfVectorPointEvaluation) {
            Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                int offset = (_lro_total_offsets[i]+0)*target_NP;
                for (int j=0; j<target_NP; ++j) {
                    P_target_row(offset + j, basis_multiplier_component) = 0;
                }

                P_target_row(offset + 1, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);

                if (_dimensions>1) {
                    offset = (_lro_total_offsets[i]+1)*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                    }
                    P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
                }

                if (_dimensions>2) {
                    offset = (_lro_total_offsets[i]+2)*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                    }
                    P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
                }
            });
        } else if (_operations(i) == TargetOperation::CurlOfVectorPointEvaluation) {
            // comments based on taking curl of vector [u_{0},u_{1},u_{2}]^T
            // with as e.g., u_{1,z} being the partial derivative with respect to z of
            // u_{1}
            if (_dimensions==3) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    // output component 0
                    // u_{2,y} - u_{1,z}
                    {
                        int offset = (_lro_total_offsets[i]+0)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 0 on component 0 of curl
                        // (no contribution)

                        offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 1 on component 0 of curl
                        // -u_{1,z}
                        P_target_row(offset + 3, basis_multiplier_component) = -std::pow(_epsilons(target_index), -1);

                        offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+0)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 2 on component 0 of curl
                        // u_{2,y}
                        P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
                    }

                    // output component 1
                    // -u_{2,x} + u_{0,z}
                    {
                        int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+1)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 0 on component 1 of curl
                        // u_{0,z}
                        P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);

                        offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+1)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 1 on component 1 of curl
                        // (no contribution)

                        offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+1)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 2 on component 1 of curl
                        // -u_{2,x}
                        P_target_row(offset + 1, basis_multiplier_component) = -std::pow(_epsilons(target_index), -1);
                    }

                    // output component 2
                    // u_{1,x} - u_{0,y}
                    {
                        int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+2)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 0 on component 1 of curl
                        // -u_{0,y}
                        P_target_row(offset + 2, basis_multiplier_component) = -std::pow(_epsilons(target_index), -1);

                        offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+2)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 1 on component 1 of curl
                        // u_{1,x}
                        P_target_row(offset + 1, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);

                        offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+2)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 2 on component 1 of curl
                        // (no contribution)
                    }
                });
            } else if (_dimensions==2) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    // output component 0
                    // u_{1,y}
                    {
                        int offset = (_lro_total_offsets[i]+0)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 0 on component 0 of curl
                        // (no contribution)

                        offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 1 on component 0 of curl
                        // -u_{1,z}
                        P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
                    }

                    // output component 1
                    // -u_{0,x}
                    {
                        int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+1)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 0 on component 1 of curl
                        // u_{0,z}
                        P_target_row(offset + 1, basis_multiplier_component) = -std::pow(_epsilons(target_index), -1);

                        offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+1)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        // role of input 1 on component 1 of curl
                        // (no contribution)
                    }
                });
            }
        } else {
            ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
        }
    }
}

KOKKOS_INLINE_FUNCTION
void GMLS::computeCurvatureFunctionals(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type P_target_row, scratch_matrix_type* V, const int basis_multiplier_component) const {

    const int target_index = teamMember.league_rank();

    const int manifold_NP = this->getNP(_curvature_poly_order, _dimensions-1);
    for (int i=0; i<_curvature_support_operations.size(); ++i) {
        if (_curvature_support_operations(i) == TargetOperation::ScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                int offset = i*manifold_NP;
                this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _curvature_poly_order, false /*bool on only specific order*/, V, SamplingFunctional::PointSample);
                for (int j=0; j<manifold_NP; ++j) {
                    P_target_row(offset + j, basis_multiplier_component) = t1(j);
                }
            });
        } else if (_curvature_support_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                int offset = i*manifold_NP;
                this->calcGradientPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions-1, _curvature_poly_order, false /*specific order only*/, V, SamplingFunctional::PointSample);
                for (int j=0; j<manifold_NP; ++j) {
                    P_target_row(offset + j, basis_multiplier_component) = t1(j);
                }
                if (_dimensions>2) { // _dimensions-1 > 1
                    offset = (i+1)*manifold_NP;
                    this->calcGradientPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions-1, _curvature_poly_order, false /*specific order only*/, V, SamplingFunctional::PointSample);
                    for (int j=0; j<manifold_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = t1(j);
                    }
                }
            });
        } else {
            ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
        }
    }
}

KOKKOS_INLINE_FUNCTION
void GMLS::computeTargetFunctionalsOnManifold(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type P_target_row, scratch_matrix_type V, scratch_matrix_type G_inv, scratch_vector_type curvature_coefficients, scratch_vector_type curvature_gradients, const int basis_multiplier_component) const {

    // only designed for 2D manifold embedded in 3D space
    const int target_index = teamMember.league_rank();
    const int target_NP = this->getNP(_poly_order, _dimensions-1);

    for (int i=0; i<_operations.size(); ++i) {
        if (_dimensions>2) {
            if (_operations(i) == TargetOperation::ScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
//                    this->calcPij(P_target_row.data()+_lro_total_offsets[i]*target_NP, target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, SamplingFunctional::PointSample);
                    this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, SamplingFunctional::PointSample);
                    int offset = _lro_total_offsets[i]*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = t1(j);
                    }
                });
            } else if (_operations(i) == TargetOperation::VectorPointEvaluation) {

                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {

                    // output component 0
                    this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, SamplingFunctional::PointSample);
                    int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = t1(j);
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }
                    offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }
                    offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }

                    // output component 1
                    offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }
                    offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j);
                    }
                    offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }

                    // output component 2
                    offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }
                    offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }
                    offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }

                });
            } else if (_operations(i) == TargetOperation::LaplacianOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {

                    double h = _epsilons(target_index);
                    double a1, a2, a3, a4, a5;
                    if (_curvature_poly_order > 0) {
                        a1 = curvature_coefficients(1);
                        a2 = curvature_coefficients(2);
                    }
                    if (_curvature_poly_order > 1) {
                        a3 = curvature_coefficients(3);
                        a4 = curvature_coefficients(4);
                        a5 = curvature_coefficients(5);
                    }
                    double den = (h*h + a1*a1 + a2*a2);

                    // Gaussian Curvature sanity check
                    //double K_curvature = ( - a4*a4 + a3*a5) / den / den;
                    //std::cout << "Gaussian curvature is: " << K_curvature << std::endl;


                    const int offset = _lro_total_offsets[i]*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                    }
                    // scaled
                    if (_poly_order > 0 && _curvature_poly_order > 1) {
                        P_target_row(offset + 1, basis_multiplier_component) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
                        P_target_row(offset + 2, basis_multiplier_component) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
                    }
                    if (_poly_order > 1 && _curvature_poly_order > 0) {
                        P_target_row(offset + 3, basis_multiplier_component) = (h*h+a2*a2)/den/(h*h);
                        P_target_row(offset + 4, basis_multiplier_component) = -2*a1*a2/den/(h*h);
                        P_target_row(offset + 5, basis_multiplier_component) = (h*h+a1*a1)/den/(h*h);
                    }

                });
            } else if (_operations(i) == TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation) {
                if (_reconstruction_space == ReconstructionSpace::VectorTaylorPolynomial) {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1, a2, a3, a4, a5;
                        if (_curvature_poly_order > 0) {
                            a1 = curvature_coefficients(1);
                            a2 = curvature_coefficients(2);
                        }
                        if (_curvature_poly_order > 1) {
                            a3 = curvature_coefficients(3);
                            a4 = curvature_coefficients(4);
                            a5 = curvature_coefficients(5);
                        }
                        double den = (h*h + a1*a1 + a2*a2);

                        double c0a = -a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5)/den/den/h;
                        double c1a = (h*h+a2*a2)/den/h;
                        double c2a = -a1*a2/den/h;

                        double c0b = -a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5)/den/den/h;
                        double c1b = -a1*a2/den/h;
                        double c2b = (h*h+a1*a1)/den/h;

                        // 1st input component
                        int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                            P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                        }
                        P_target_row(offset + 0, basis_multiplier_component) = c0a;
                        P_target_row(offset + 1, basis_multiplier_component) = c1a;
                        P_target_row(offset + 2, basis_multiplier_component) = c2a;
                        P_target_row(offset + target_NP + 0, basis_multiplier_component) = c0b;
                        P_target_row(offset + target_NP + 1, basis_multiplier_component) = c1b;
                        P_target_row(offset + target_NP + 2, basis_multiplier_component) = c2b;
                    });
                } else if (_reconstruction_space == ReconstructionSpace::ScalarTaylorPolynomial) {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1, a2, a3, a4, a5;
                        if (_curvature_poly_order > 0) {
                            a1 = curvature_coefficients(1);
                            a2 = curvature_coefficients(2);
                        }
                        if (_curvature_poly_order > 1) {
                            a3 = curvature_coefficients(3);
                            a4 = curvature_coefficients(4);
                            a5 = curvature_coefficients(5);
                        }
                        double den = (h*h + a1*a1 + a2*a2);

                        const int offset = _lro_total_offsets[i]*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }

                        // verified
                        if (_poly_order > 0 && _curvature_poly_order > 1) {
                            P_target_row(offset + 1, basis_multiplier_component) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
                            P_target_row(offset + 2, basis_multiplier_component) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
                        }
                        if (_poly_order > 1 && _curvature_poly_order > 0) {
                            P_target_row(offset + 3, basis_multiplier_component) = (h*h+a2*a2)/den/(h*h);
                            P_target_row(offset + 4, basis_multiplier_component) = -2*a1*a2/den/(h*h);
                            P_target_row(offset + 5, basis_multiplier_component) = (h*h+a1*a1)/den/(h*h);
                        }

                    });
                }
            } else if (_operations(i) == TargetOperation::LaplacianOfVectorPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {

                    double epsilons_target_index = _epsilons(target_index);

                    double h = _epsilons(target_index);
                    double a1, a2, a3, a4, a5;
                    if (_curvature_poly_order > 0) {
                        a1 = curvature_coefficients(1);
                        a2 = curvature_coefficients(2);
                    }
                    if (_curvature_poly_order > 1) {
                        a3 = curvature_coefficients(3);
                        a4 = curvature_coefficients(4);
                        a5 = curvature_coefficients(5);
                    }
                    double den = (h*h + a1*a1 + a2*a2);

                    for (int j=0; j<target_NP; ++j) {
                        t1(j) = 0;
                    }

                    // 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*Inv[G]*P
                    if (_poly_order > 0 && _curvature_poly_order > 1) {
                        t1(1) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4+(h*h+a1*a1)*a5))/den/den/(h*h);
                        t1(2) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4+(h*h+a1*a1)*a5))/den/den/(h*h);
                    }
                    if (_poly_order > 1 && _curvature_poly_order > 0) {
                        t1(3) = (h*h+a2*a2)/den/(h*h);
                        t1(4) = -2*a1*a2/den/(h*h);
                        t1(5) = (h*h+a1*a1)/den/(h*h);
                    }

                    // output component 0
                    int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = t1(j)*V(0,0);
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }
                    offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j)*V(1,0);
                    }
                    offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }

                    // output component 1
                    offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = t1(j)*V(0,1);
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }
                    offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j)*V(1,1);
                    }
                    offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }

                    // output component 2
                    offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = t1(j)*V(0,2);
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }
                    offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j)*V(1,2);
                    }
                    offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                    }

                });
            } else if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                if (_polynomial_sampling_functional==SamplingFunctional::StaggeredEdgeIntegralSample) {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                        this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, SamplingFunctional::PointSample);
                    });
                    teamMember.team_barrier();
                    int offset;
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                        for (int j=0; j<target_NP; ++j) {
                            double t1a = G_inv(0,0);
                            double t2a = G_inv(0,1);
                            double t3a = G_inv(0,0)*curvature_gradients(0) + G_inv(0,1)*curvature_gradients(1);

                            double t1b = G_inv(0,1);
                            double t2b = G_inv(1,1);
                            double t3b = G_inv(0,1)*curvature_gradients(0) + G_inv(1,1)*curvature_gradients(1);

                            double v1c = V(0,0)*t1a + V(1,0)*t2a + V(2,0)*t3a;
                            double v2c = V(0,1)*t1a + V(1,1)*t2a + V(2,1)*t3a;
                            double v3c = V(0,2)*t1a + V(1,2)*t2a + V(2,2)*t3a;

                            double v1d = V(0,0)*t1b + V(1,0)*t2b + V(2,0)*t3b;
                            double v2d = V(0,1)*t1b + V(1,1)*t2b + V(2,1)*t3b;
                            double v3d = V(0,2)*t1b + V(1,2)*t2b + V(2,2)*t3b;

                            offset = (_lro_total_offsets[i]+0)*_basis_multiplier*target_NP;
                            P_target_row(offset + j, basis_multiplier_component) = v1c*t1(j);
                            P_target_row(offset + target_NP + j, basis_multiplier_component) = v1d*t1(j);

                            offset = (_lro_total_offsets[i]+1)*_basis_multiplier*target_NP;
                            P_target_row(offset + j, basis_multiplier_component) = v2c*t1(j);
                            P_target_row(offset + target_NP + j, basis_multiplier_component) = v2d*t1(j);

                            offset = (_lro_total_offsets[i]+2)*_basis_multiplier*target_NP;
                            P_target_row(offset + j, basis_multiplier_component) = v3c*t1(j);
                            P_target_row(offset + target_NP + j, basis_multiplier_component) = v3d*t1(j);
                        }
                    });
                } else {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1 = curvature_coefficients(1);
                        double a2 = curvature_coefficients(2);

                        double q1 = (h*h + a2*a2)/(h*h*h + h*a1*a1 + h*a2*a2);
                        double q2 = -(a1*a2)/(h*h*h + h*a1*a1 + h*a2*a2);
                        double q3 = (h*h + a1*a1)/(h*h*h + h*a1*a1 + h*a2*a2);

                        double t1a = q1*1 + q2*0;
                        double t2a = q1*0 + q2*1;

                        double t1b = q2*1 + q3*0;
                        double t2b = q2*0 + q3*1;

                        // scaled
                        int offset = (_lro_total_offsets[i]+0)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        if (_poly_order > 0 && _curvature_poly_order > 0) {
                            P_target_row(offset + 1, basis_multiplier_component) = t1a + t2a;
                            P_target_row(offset + 2, basis_multiplier_component) = 0;
                        }

                        offset = (_lro_total_offsets[i]+1)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        if (_poly_order > 0 && _curvature_poly_order > 0) {
                            P_target_row(offset + 1, basis_multiplier_component) = 0;
                            P_target_row(offset + 2, basis_multiplier_component) = t1b + t2b;
                        }

                        offset = (_lro_total_offsets[i]+2)*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                        }
                        if (_poly_order > 0 && _curvature_poly_order > 0) {
                            P_target_row(offset + 1, basis_multiplier_component) = 0;
                            P_target_row(offset + 2, basis_multiplier_component) = 0;
                        }

                    });
                }
            } else if (_operations(i) == TargetOperation::PartialXOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = _lro_total_offsets[i]*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
                        t2(j) = (j == 2) ? std::pow(_epsilons(target_index), -1) : 0;
                    }
                    for (int j=0; j<target_NP; ++j) {
                        double v1 =    t1(j)*G_inv(0,0) + t2(j)*G_inv(1,0);
                        double v2 =    t1(j)*G_inv(0,1) + t2(j)*G_inv(1,1);

                        double t1 = v1*1 + v2*0;
                        double t2 = v1*0 + v2*1;
                        double t3 = v1*curvature_gradients(0) + v2*curvature_gradients(1);

                        P_target_row(offset + j, basis_multiplier_component) = t1*V(0,0) + t2*V(1,0) + t3*V(2,0);
                    }
                });
            } else if (_operations(i) == TargetOperation::PartialYOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = _lro_total_offsets[i]*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
                        t2(j) = (j == 2) ? std::pow(_epsilons(target_index), -1) : 0;
                    }

                    for (int j=0; j<target_NP; ++j) {
                        double v1 =    t1(j)*G_inv(0,0) + t2(j)*G_inv(1,0);
                        double v2 =    t1(j)*G_inv(0,1) + t2(j)*G_inv(1,1);

                        double t1 = v1*1 + v2*0;
                        double t2 = v1*0 + v2*1;
                        double t3 = v1*curvature_gradients(0) + v2*curvature_gradients(1);

                        P_target_row(offset + j, basis_multiplier_component) = t1*V(0,1) + t2*V(1,1) + t3*V(2,1);
                    }
                });
            } else if (_operations(i) == TargetOperation::PartialZOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = _lro_total_offsets[i]*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
                        t2(j) = (j == 2) ? std::pow(_epsilons(target_index), -1) : 0;
                    }

                    for (int j=0; j<target_NP; ++j) {
                        double v1 =    t1(j)*G_inv(0,0) + t2(j)*G_inv(1,0);
                        double v2 =    t1(j)*G_inv(0,1) + t2(j)*G_inv(1,1);

                        double t1 = v1*1 + v2*0;
                        double t2 = v1*0 + v2*1;
                        double t3 = v1*curvature_gradients(0) + v2*curvature_gradients(1);

                        P_target_row(offset + j, basis_multiplier_component) = t1*V(0,2) + t2*V(1,2) + t3*V(2,2);
                    }
                });
            } else if (_operations(i) == TargetOperation::DivergenceOfVectorPointEvaluation) {
                if (_data_sampling_functional==SamplingFunctional::StaggeredEdgeIntegralSample) {
                    ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
                } else if (_data_sampling_functional==SamplingFunctional::ManifoldGradientVectorSample || _data_sampling_functional==SamplingFunctional::ManifoldVectorSample) {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1, a2, a3, a4, a5;
                        if (_curvature_poly_order > 0) {
                            a1 = curvature_coefficients(1);
                            a2 = curvature_coefficients(2);
                        }
                        if (_curvature_poly_order > 1) {
                            a3 = curvature_coefficients(3);
                            a4 = curvature_coefficients(4);
                            a5 = curvature_coefficients(5);
                        }
                        double den = (h*h + a1*a1 + a2*a2);

                        // 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*P
                        // i.e. P recovers G^{-1}*grad of scalar
                        double c0a = (a1*a3+a2*a4)/(h*den);
                        double c1a = 1./h;
                        double c2a = 0;

                        double c0b = (a1*a4+a2*a5)/(h*den);
                        double c1b = 0;
                        double c2b = 1./h;

                        // 1st input component
                        int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                            P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                        }
                        P_target_row(offset + 0, basis_multiplier_component) = c0a;
                        P_target_row(offset + 1, basis_multiplier_component) = c1a;
                        P_target_row(offset + 2, basis_multiplier_component) = c2a;

                        // 2nd input component
                        offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                            P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                        }
                        P_target_row(offset + target_NP + 0, basis_multiplier_component) = c0b;
                        P_target_row(offset + target_NP + 1, basis_multiplier_component) = c1b;
                        P_target_row(offset + target_NP + 2, basis_multiplier_component) = c2b;

                        //// 3rd input component (no contribution)
                        //offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                        //for (int j=0; j<target_NP; ++j) {
                        //    P_target_row(offset + j, basis_multiplier_component) = 0;
                        //    P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                        //}
                    });
                }
            } else if (_operations(i) == TargetOperation::DivergenceOfScalarPointEvaluation) {
                if (_data_sampling_functional==SamplingFunctional::StaggeredEdgeAnalyticGradientIntegralSample) {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1, a2, a3, a4, a5;
                        if (_curvature_poly_order > 0) {
                            a1 = curvature_coefficients(1);
                            a2 = curvature_coefficients(2);
                        }
                        if (_curvature_poly_order > 1) {
                            a3 = curvature_coefficients(3);
                            a4 = curvature_coefficients(4);
                            a5 = curvature_coefficients(5);
                        }
                        double den = (h*h + a1*a1 + a2*a2);

                        // 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*Inv[G].P
                        // i.e. P recovers grad of scalar
                        double c0a = -a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5)/den/den/h;
                        double c1a = (h*h+a2*a2)/den/h;
                        double c2a = -a1*a2/den/h;

                        double c0b = -a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5)/den/den/h;
                        double c1b = -a1*a2/den/h;
                        double c2b = (h*h+a1*a1)/den/h;

                        // 1st input component
                        int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                            P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                        }
                        P_target_row(offset + 0, basis_multiplier_component) = c0a;
                        P_target_row(offset + 1, basis_multiplier_component) = c1a;
                        P_target_row(offset + 2, basis_multiplier_component) = c2a;
                        P_target_row(offset + target_NP + 0, basis_multiplier_component) = c0b;
                        P_target_row(offset + target_NP + 1, basis_multiplier_component) = c1b;
                        P_target_row(offset + target_NP + 2, basis_multiplier_component) = c2b;

                    });
                } else if (_data_sampling_functional==SamplingFunctional::StaggeredEdgeIntegralSample) {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1, a2, a3, a4, a5;
                        if (_curvature_poly_order > 0) {
                            a1 = curvature_coefficients(1);
                            a2 = curvature_coefficients(2);
                        }
                        if (_curvature_poly_order > 1) {
                            a3 = curvature_coefficients(3);
                            a4 = curvature_coefficients(4);
                            a5 = curvature_coefficients(5);
                        }
                        double den = (h*h + a1*a1 + a2*a2);

                        // 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*.P
                        // i.e. P recovers G^{-1} * grad of scalar
                        double c0a = (a1*a3+a2*a4)/(h*den);
                        double c1a = 1./h;
                        double c2a = 0;

                        double c0b = (a1*a4+a2*a5)/(h*den);
                        double c1b = 0;
                        double c2b = 1./h;

                        // 1st input component
                        int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset + j, basis_multiplier_component) = 0;
                            P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
                        }
                        P_target_row(offset + 0, basis_multiplier_component) = c0a;
                        P_target_row(offset + 1, basis_multiplier_component) = c1a;
                        P_target_row(offset + 2, basis_multiplier_component) = c2a;
                        P_target_row(offset + target_NP + 0, basis_multiplier_component) = c0b;
                        P_target_row(offset + target_NP + 1, basis_multiplier_component) = c1b;
                        P_target_row(offset + target_NP + 2, basis_multiplier_component) = c2b;

                    });
                }
            }
            else {
                ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
            }
        } else if (_dimensions==2) { // 1D manifold in 2D problem
            if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = (_lro_total_offsets[i]+0)*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
                    }
                    for (int j=0; j<target_NP; ++j) {
                        double v1 =    t1(j)*G_inv(0,0);

                        double t1 = v1*1;
                        double t2 = v1*curvature_gradients(0);

                        P_target_row(offset + j, basis_multiplier_component) = t1*V(0,0) + t2*V(1,0);
                    }

                    offset = (_lro_total_offsets[i]+1)*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
                    }

                    for (int j=0; j<target_NP; ++j) {
                        double v1 =    t1(j)*G_inv(0,0);

                        double t1 = v1*1;
                        double t2 = v1*curvature_gradients(0);

                        P_target_row(offset + j, basis_multiplier_component) = t1*V(0,1) + t2*V(1,1);
                    }
                });
            } else if (_operations(i) == TargetOperation::PartialXOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = _lro_total_offsets[i]*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
                    }
                    for (int j=0; j<target_NP; ++j) {
                        double v1 =    t1(j)*G_inv(0,0);

                        double t1 = v1*1;
                        double t2 = v1*curvature_gradients(0);

                        P_target_row(offset + j, basis_multiplier_component) = t1*V(0,0) + t2*V(1,0);
                    }
                });
            } else if (_operations(i) == TargetOperation::PartialYOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = _lro_total_offsets[i]*target_NP;
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset + j, basis_multiplier_component) = 0;
                        t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
                    }

                    for (int j=0; j<target_NP; ++j) {
                        double v1 =    t1(j)*G_inv(0,0);

                        double t1 = v1*1;
                        double t2 = v1*curvature_gradients(0);

                        P_target_row(offset + j, basis_multiplier_component) = t1*V(0,1) + t2*V(1,1);
                    }
                });
            } else {
                ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
            }
        }
    }
}

}; // Compadre
#endif
