#ifndef _COMPADRE_GMLS_TARGETS_HPP_
#define _COMPADRE_GMLS_TARGETS_HPP_

#include "Compadre_GMLS.hpp"
#include "Compadre_Manifold_Functions.hpp"
#include "basis/DivergenceFree3D.hpp"

namespace Compadre {

KOKKOS_INLINE_FUNCTION
void GMLS::computeTargetFunctionals(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_right_type P_target_row) const {

    // check if VectorOfScalarClonesTaylorPolynomial is used with a scalar sampling functional other than PointSample
    if (_dimensions > 1) {
            compadre_kernel_assert_debug(
            (_reconstruction_space!=ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial
                || _data_sampling_multiplier!=0
                || (_reconstruction_space==ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial 
                        && _polynomial_sampling_functional==PointSample))
            && "_reconstruction_space(VectorOfScalar clones incompatible with scalar output sampling functional which is not a PointSample");
    } 

    // determine if additional evaluation sites are requested by user and handled by target operations 
    bool additional_evaluation_sites_need_handled = 
        (_additional_evaluation_coordinates.extent(0) > 0) ? true : false; // additional evaluation sites are specified

    const int target_index = _initial_index_for_batch + teamMember.league_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, P_target_row.extent(0)), [&] (const int j) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, P_target_row.extent(1)),
          [=] (const int& k) {
            P_target_row(j,k) = 0;
        });
    });
    teamMember.team_barrier();

    const int target_NP = this->getNP(_poly_order, _dimensions, _reconstruction_space);
    const int num_evaluation_sites = (static_cast<int>(_additional_evaluation_indices.extent(1)) > 1) 
                ? static_cast<int>(getNAdditionalEvaluationCoordinates(target_index)+1) : 1;

    for (size_t i=0; i<_operations.size(); ++i) {

        bool additional_evaluation_sites_handled = false; // target operations that can handle these sites should flip this flag

        bool operation_handled = true;

        // USER defined targets should be added to this file
        // if the USER defined targets don't catch this operation, then operation_handled will be false
        #include "USER_StandardTargetFunctionals.hpp"

        // if the user didn't handle the operation, we pass it along to the toolkit's targets
        if (!operation_handled) {

        if (_reconstruction_space == ReconstructionSpace::ScalarTaylorPolynomial) {

            /*
             * Beginning of ScalarTaylorPolynomial basis
             */

            if (_operations(i) == TargetOperation::ScalarPointEvaluation || (_operations(i) == TargetOperation::VectorPointEvaluation && _dimensions == 1) /* vector is a scalar in 1D */) {
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        for (int k=0; k<target_NP; ++k) {
                            P_target_row(offset, k) = t1(k);
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::LaplacianOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                    switch (_dimensions) {
                    case 3:
                        P_target_row(offset, 4) = std::pow(_epsilons(target_index), -2);
                        P_target_row(offset, 6) = std::pow(_epsilons(target_index), -2);
                        P_target_row(offset, 9) = std::pow(_epsilons(target_index), -2);
                        break;
                    case 2:
                        P_target_row(offset, 3) = std::pow(_epsilons(target_index), -2);
                        P_target_row(offset, 5) = std::pow(_epsilons(target_index), -2);
                        break;
                    default:
                        P_target_row(offset, 2) = std::pow(_epsilons(target_index), -2);
                    }
                });
            } else if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        for (int d=0; d<_dimensions; ++d) {
                            int offset = getTargetOffsetIndexDevice(i, 0, d, j);
                            auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                            this->calcGradientPij(row.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, d /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialXOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                        this->calcGradientPij(row.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialYOfScalarPointEvaluation) {
                compadre_kernel_assert_release(_dimensions>1 && "PartialYOfScalarPointEvaluation requested for dim < 2");
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                        this->calcGradientPij(row.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialZOfScalarPointEvaluation) {
                compadre_kernel_assert_release(_dimensions>2 && "PartialZOfScalarPointEvaluation requested for dim < 3");
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                        this->calcGradientPij(row.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            }
            // staggered gradient w/ edge integrals known analytically, using a basis
            // of potentials
            else if (_operations(i) == TargetOperation::DivergenceOfVectorPointEvaluation
                     && _polynomial_sampling_functional == StaggeredEdgeAnalyticGradientIntegralSample) {
              Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                  int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                  switch (_dimensions) {
                  case 3:
                      P_target_row(offset, 4) = std::pow(_epsilons(target_index), -2);
                      P_target_row(offset, 6) = std::pow(_epsilons(target_index), -2);
                      P_target_row(offset, 9) = std::pow(_epsilons(target_index), -2);
                      break;
                  case 2:
                      P_target_row(offset, 3) = std::pow(_epsilons(target_index), -2);
                      P_target_row(offset, 5) = std::pow(_epsilons(target_index), -2);
                      break;
                  default:
                      P_target_row(offset, 2) = std::pow(_epsilons(target_index), -2);
                  }
              });
            }
            else {
                compadre_kernel_assert_release((false) && "Functionality not yet available.");
            }

            /*
             * End of ScalarTaylorPolynomial basis
             */

        } else if (_reconstruction_space == ReconstructionSpace::VectorTaylorPolynomial) {

            /*
             * Beginning of VectorTaylorPolynomial basis
             */

            if (_operations(i) == TargetOperation::ScalarPointEvaluation || (_operations(i) == TargetOperation::VectorPointEvaluation && _dimensions == 1) /* vector is a scalar in 1D */) {
                // copied from ScalarTaylorPolynomial
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        for (int k=0; k<target_NP; ++k) {
                            P_target_row(offset, k) = t1(k);
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::VectorPointEvaluation) {
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int e=0; e<num_evaluation_sites; ++e) {
                        this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, e);
                        for (int m=0; m<_sampling_multiplier; ++m) {
                            int output_components = _basis_multiplier;
                            for (int c=0; c<output_components; ++c) {
                                int offset = getTargetOffsetIndexDevice(i, m /*in*/, c /*out*/, e/*additional*/);
                                // for the case where _sampling_multiplier is > 1,
                                // this approach relies on c*target_NP being equivalent to P_target_row(offset, j) where offset is 
                                // getTargetOffsetIndexDevice(i, m /*in*/, c /*out*/, e/*additional*/)*_basis_multiplier*target_NP;
                                for (int j=0; j<target_NP; ++j) {
                                    P_target_row(offset, c*target_NP + j) = t1(j);
                                }
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::DivergenceOfVectorPointEvaluation) {
                if (_polynomial_sampling_functional == StaggeredEdgeIntegralSample) {
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);;
                        switch (_dimensions) {
                        case 3:
                            P_target_row(offset, 1) = std::pow(_epsilons(target_index), -1);
                            P_target_row(offset, target_NP + 2) = std::pow(_epsilons(target_index), -1);
                            P_target_row(offset, 2*target_NP + 3) = std::pow(_epsilons(target_index), -1);
                            break;
                        case 2:
                            P_target_row(offset, 1) = std::pow(_epsilons(target_index), -1);
                            P_target_row(offset, target_NP + 2) = std::pow(_epsilons(target_index), -1);
                            break;
                        default:
                            P_target_row(offset, 1) = std::pow(_epsilons(target_index), -1);
                        }
                    });
                } else {
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                      for (int e=0; e<num_evaluation_sites; ++e) {
                          for (int m=0; m<_sampling_multiplier; ++m) {
                              this->calcGradientPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, m /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, e);
                              int offset = getTargetOffsetIndexDevice(i, m /*in*/, 0 /*out*/, e/*additional*/);
                              for (int j=0; j<target_NP; ++j) {
                                  P_target_row(offset, m*target_NP + j) = t1(j);
                              }
                          }
                      }
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                }
            } else if (_operations(i) == TargetOperation::CurlOfVectorPointEvaluation) {
                if (_dimensions==3) { 
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                        // output component 0
                        // u_{2,y} - u_{1,z}
                        {
                            int offset = getTargetOffsetIndexDevice(i, 0 /*in*/, 0 /*out*/, 0/*additional*/);
                            // role of input 0 on component 0 of curl
                            // (no contribution)

                            offset = getTargetOffsetIndexDevice(i, 1 /*in*/, 0 /*out*/, 0/*additional*/);
                            // role of input 1 on component 0 of curl
                            // -u_{1,z}
                            P_target_row(offset, target_NP + 3) = -std::pow(_epsilons(target_index), -1);

                            offset = getTargetOffsetIndexDevice(i, 2 /*in*/, 0 /*out*/, 0/*additional*/);
                            // role of input 2 on component 0 of curl
                            // u_{2,y}
                            P_target_row(offset, 2*target_NP + 2) = std::pow(_epsilons(target_index), -1);
                        }

                        // output component 1
                        // -u_{2,x} + u_{0,z}
                        {
                            int offset = getTargetOffsetIndexDevice(i, 0 /*in*/, 1 /*out*/, 0/*additional*/);
                            // role of input 0 on component 1 of curl
                            // u_{0,z}
                            P_target_row(offset, 3) = std::pow(_epsilons(target_index), -1);

                            // offset = getTargetOffsetIndexDevice(i, 1 /*in*/, 1 /*out*/, 0/*additional*/);
                            // role of input 1 on component 1 of curl
                            // (no contribution)

                            offset = getTargetOffsetIndexDevice(i, 2 /*in*/, 1 /*out*/, 0/*additional*/);
                            // role of input 2 on component 1 of curl
                            // -u_{2,x}
                            P_target_row(offset, 2*target_NP + 1) = -std::pow(_epsilons(target_index), -1);
                        }

                        // output component 2
                        // u_{1,x} - u_{0,y}
                        {
                            int offset = getTargetOffsetIndexDevice(i, 0 /*in*/, 2 /*out*/, 0/*additional*/);
                            // role of input 0 on component 1 of curl
                            // -u_{0,y}
                            P_target_row(offset, 2) = -std::pow(_epsilons(target_index), -1);

                            offset = getTargetOffsetIndexDevice(i, 1 /*in*/, 2 /*out*/, 0/*additional*/);
                            // role of input 1 on component 1 of curl
                            // u_{1,x}
                            P_target_row(offset, target_NP + 1) = std::pow(_epsilons(target_index), -1);

                            // offset = getTargetOffsetIndexDevice(i, 2 /*in*/, 2 /*out*/, 0/*additional*/);
                            // role of input 2 on component 1 of curl
                            // (no contribution)
                        }
                    });
                } else if (_dimensions==2) {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                        // output component 0
                        // u_{1,y}
                        {
                            int offset = getTargetOffsetIndexDevice(i, 0 /*in*/, 0 /*out*/, 0/*additional*/);
                            // role of input 0 on component 0 of curl
                            // (no contribution)

                            offset = getTargetOffsetIndexDevice(i, 1 /*in*/, 0 /*out*/, 0/*additional*/);
                            // role of input 1 on component 0 of curl
                            // -u_{1,z}
                            P_target_row(offset, target_NP + 2) = std::pow(_epsilons(target_index), -1);
                        }

                        // output component 1
                        // -u_{0,x}
                        {
                            int offset = getTargetOffsetIndexDevice(i, 0 /*in*/, 1 /*out*/, 0/*additional*/);
                            // role of input 0 on component 1 of curl
                            // u_{0,z}
                            P_target_row(offset, 1) = -std::pow(_epsilons(target_index), -1);

                            //offset = getTargetOffsetIndexDevice(i, 1 /*in*/, 1 /*out*/, 0/*additional*/);
                            // role of input 1 on component 1 of curl
                            // (no contribution)
                        }
                    });
                }
            } else {
                compadre_kernel_assert_release((false) && "Functionality not yet available.");
            }

            /*
             * End of VectorTaylorPolynomial basis
             */

        } else if (_reconstruction_space == ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial) {

            /*
             * Beginning of VectorOfScalarClonesTaylorPolynomial basis
             */

            if (_operations(i) == TargetOperation::ScalarPointEvaluation || (_operations(i) == TargetOperation::VectorPointEvaluation && _dimensions == 1) /* vector is a scalar in 1D */) {
                // copied from ScalarTaylorPolynomial
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        for (int k=0; k<target_NP; ++k) {
                            P_target_row(offset, k) = t1(k);
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::VectorPointEvaluation) {
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int e=0; e<num_evaluation_sites; ++e) {
                        this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, e);
                        for (int m=0; m<_sampling_multiplier; ++m) {
                            for (int c=0; c<_data_sampling_multiplier; ++c) {
                                int offset = getTargetOffsetIndexDevice(i, c /*in*/, c /*out*/, e/*additional*/);
                                for (int j=0; j<target_NP; ++j) {
                                    P_target_row(offset, j) = t1(j);
                                }
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::LaplacianOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                    switch (_dimensions) {
                    case 3:
                        P_target_row(offset, 4) = std::pow(_epsilons(target_index), -2);
                        P_target_row(offset, 6) = std::pow(_epsilons(target_index), -2);
                        P_target_row(offset, 9) = std::pow(_epsilons(target_index), -2);
                        break;
                    case 2:
                        P_target_row(offset, 3) = std::pow(_epsilons(target_index), -2);
                        P_target_row(offset, 5) = std::pow(_epsilons(target_index), -2);
                        break;
                    default:
                        P_target_row(offset, 2) = std::pow(_epsilons(target_index), -2);
                    }
                });
            } else if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        for (int d=0; d<_dimensions; ++d) {
                            int offset = getTargetOffsetIndexDevice(i, 0, d, j);
                            auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                            this->calcGradientPij(row.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, d /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialXOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                        this->calcGradientPij(row.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialYOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
                if (_dimensions>1) {
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                        for (int j=0; j<num_evaluation_sites; ++j) { 
                            int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                            auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                            this->calcGradientPij(row.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                        }
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                }
            } else if (_operations(i) == TargetOperation::PartialZOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
                if (_dimensions>2) {
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                        for (int j=0; j<num_evaluation_sites; ++j) { 
                            int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                            auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                            this->calcGradientPij(row.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                        }
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                }
            } else if (_operations(i) == TargetOperation::DivergenceOfVectorPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset, j) = 0;
                    }

                    P_target_row(offset, 1) = std::pow(_epsilons(target_index), -1);

                    if (_dimensions>1) {
                        offset = getTargetOffsetIndexDevice(i, 1, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                        }
                        P_target_row(offset, 2) = std::pow(_epsilons(target_index), -1);
                    }

                    if (_dimensions>2) {
                        offset = getTargetOffsetIndexDevice(i, 2, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                        }
                        P_target_row(offset, 3) = std::pow(_epsilons(target_index), -1);
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
                            int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 0 on component 0 of curl
                            // (no contribution)

                            offset = getTargetOffsetIndexDevice(i, 1, 0, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 1 on component 0 of curl
                            // -u_{1,z}
                            P_target_row(offset, 3) = -std::pow(_epsilons(target_index), -1);

                            offset = getTargetOffsetIndexDevice(i, 2, 0, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 2 on component 0 of curl
                            // u_{2,y}
                            P_target_row(offset, 2) = std::pow(_epsilons(target_index), -1);
                        }

                        // output component 1
                        // -u_{2,x} + u_{0,z}
                        {
                            int offset = getTargetOffsetIndexDevice(i, 0, 1, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 0 on component 1 of curl
                            // u_{0,z}
                            P_target_row(offset, 3) = std::pow(_epsilons(target_index), -1);

                            offset = getTargetOffsetIndexDevice(i, 1, 1, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 1 on component 1 of curl
                            // (no contribution)

                            offset = getTargetOffsetIndexDevice(i, 2, 1, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 2 on component 1 of curl
                            // -u_{2,x}
                            P_target_row(offset, 1) = -std::pow(_epsilons(target_index), -1);
                        }

                        // output component 2
                        // u_{1,x} - u_{0,y}
                        {
                            int offset = getTargetOffsetIndexDevice(i, 0, 2, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 0 on component 1 of curl
                            // -u_{0,y}
                            P_target_row(offset, 2) = -std::pow(_epsilons(target_index), -1);

                            offset = getTargetOffsetIndexDevice(i, 1, 2, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 1 on component 1 of curl
                            // u_{1,x}
                            P_target_row(offset, 1) = std::pow(_epsilons(target_index), -1);

                            offset = getTargetOffsetIndexDevice(i, 2, 2, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
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
                            int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 0 on component 0 of curl
                            // (no contribution)

                            offset = getTargetOffsetIndexDevice(i, 1, 0, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 1 on component 0 of curl
                            // -u_{1,z}
                            P_target_row(offset, 2) = std::pow(_epsilons(target_index), -1);
                        }

                        // output component 1
                        // -u_{0,x}
                        {
                            int offset = getTargetOffsetIndexDevice(i, 0, 1, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 0 on component 1 of curl
                            // u_{0,z}
                            P_target_row(offset, 1) = -std::pow(_epsilons(target_index), -1);

                            offset = getTargetOffsetIndexDevice(i, 1, 1, 0);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            // role of input 1 on component 1 of curl
                            // (no contribution)
                        }
                    });
                }
            } else {
                compadre_kernel_assert_release((false) && "Functionality not yet available.");
            }

            /*
             * End of VectorOfScalarClonesTaylorPolynomial basis
             */

        } else if (_reconstruction_space == ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial) {

            /*
             * Beginning of DivergenceFreeVectorTaylorPolynomial basis
             */

            if (_operations(i) == TargetOperation::VectorPointEvaluation) {
                // copied from VectorTaylorPolynomial
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int m0=0; m0<_sampling_multiplier; ++m0) {
                        for (int m1=0; m1<_sampling_multiplier; ++m1) {

                          this->calcPij(t1.data(), target_index, -(m1+1) /* target is neighbor, but also which component */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, 0 /* evaluate at target */);

                          int offset = getTargetOffsetIndexDevice(i, m0 /*in*/, m1 /*out*/, 0 /*no additional*/);
                          for (int j=0; j<target_NP; ++j) {
                              P_target_row(offset, j) = t1(j);
                          }
                        }
                    }
                });
            } else if (_operations(i) == TargetOperation::CurlOfVectorPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    for (int m0=0; m0<_sampling_multiplier; ++m0) { // input components
                        for (int m1=0; m1<_sampling_multiplier; ++m1) { // output components
                            int offset = getTargetOffsetIndexDevice(i, m0 /*in*/, m1 /*out*/, 0 /*no additional*/);
                            switch (m1) {
                                // manually compute the output components
                                case 0:
                                    // output component 0
                                    P_target_row(offset, 6) = -std::pow(_epsilons(target_index), -1);
                                    P_target_row(offset, 8) = std::pow(_epsilons(target_index), -1);
                                    break;
                                case 1:
                                    // output component 1
                                    P_target_row(offset, 7) = -std::pow(_epsilons(target_index), -1);
                                    P_target_row(offset, 4) = std::pow(_epsilons(target_index), -1);
                                    break;
                                default:
                                    // output component 2
                                    P_target_row(offset, 3) = -std::pow(_epsilons(target_index), -1);
                                    P_target_row(offset, 5) = std::pow(_epsilons(target_index), -1);
                                    break;
                            }
                        }
                    }
                });
            } else if (_operations(i) == TargetOperation::CurlCurlOfVectorPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    for (int m0=0; m0<_sampling_multiplier; ++m0) { // input components
                        for (int m1=0; m1<_sampling_multiplier; ++m1) { // output components
                            int offset = getTargetOffsetIndexDevice(i, m0 /*in*/, m1 /*out*/, 0 /*no additional*/);
                            switch (m1) {
                                // manually compute the output components
                                case 0:
                                    // output component 0
                                    P_target_row(offset, 11) = -2.0*std::pow(_epsilons(target_index), -2);
                                    P_target_row(offset, 12) = -2.0*std::pow(_epsilons(target_index), -2);
                                    P_target_row(offset, 20) = -2.0*std::pow(_epsilons(target_index), -2);
                                    break;
                                case 1:
                                    // output component 1
                                    P_target_row(offset, 14) = -2.0*std::pow(_epsilons(target_index), -2);
                                    P_target_row(offset, 15) = -2.0*std::pow(_epsilons(target_index), -2);
                                    P_target_row(offset, 23) = -2.0*std::pow(_epsilons(target_index), -2);
                                    break;
                                default:
                                    // output component 2
                                    P_target_row(offset, 17) = -2.0*std::pow(_epsilons(target_index), -2);
                                    P_target_row(offset, 18) = -2.0*std::pow(_epsilons(target_index), -2);
                                    P_target_row(offset, 25) = -2.0*std::pow(_epsilons(target_index), -2);
                                    break;
                            }
                        }
                    }
                });
            }
            additional_evaluation_sites_handled = false; // additional non-target site evaluations handled
        } else {
          compadre_kernel_assert_release((false) && "Functionality not yet available.");
        }

        /*
         * End of DivergenceFreeVectorTaylorPolynomial basis
         */
        compadre_kernel_assert_release(((additional_evaluation_sites_need_handled && additional_evaluation_sites_handled) || (!additional_evaluation_sites_need_handled)) && "Auxiliary evaluation coordinates are specified by user, but are calling a target operation that can not handle evaluating additional sites.");
        } // !operation_handled
        teamMember.team_barrier();
    }
}

KOKKOS_INLINE_FUNCTION
void GMLS::computeCurvatureFunctionals(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_right_type P_target_row, const scratch_matrix_right_type* V, const local_index_type local_neighbor_index) const {

    const int target_index = _initial_index_for_batch + teamMember.league_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, P_target_row.extent(0)), [&] (const int j) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, P_target_row.extent(1)),
          [=] (const int& k) {
            P_target_row(j,k) = 0;
        });
    });
    teamMember.team_barrier();

    const int manifold_NP = this->getNP(_curvature_poly_order, _dimensions-1, ReconstructionSpace::ScalarTaylorPolynomial);
    for (size_t i=0; i<_curvature_support_operations.size(); ++i) {
        if (_curvature_support_operations(i) == TargetOperation::ScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                this->calcPij(t1.data(), target_index, local_neighbor_index, 0 /*alpha*/, _dimensions-1, _curvature_poly_order, false /*bool on only specific order*/, V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                for (int j=0; j<manifold_NP; ++j) {
                    P_target_row(offset, j) = t1(j);
                }
            });
        } else if (_curvature_support_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                //int offset = i*manifold_NP;
                int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                this->calcGradientPij(t1.data(), target_index, local_neighbor_index, 0 /*alpha*/, 0 /*partial_direction*/, _dimensions-1, _curvature_poly_order, false /*specific order only*/, V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                for (int j=0; j<manifold_NP; ++j) {
                    P_target_row(offset, j) = t1(j);
                }
                if (_dimensions>2) { // _dimensions-1 > 1
                    //offset = (i+1)*manifold_NP;
                    offset = getTargetOffsetIndexDevice(i, 0, 1, 0);
                    this->calcGradientPij(t1.data(), target_index, local_neighbor_index, 0 /*alpha*/, 1 /*partial_direction*/, _dimensions-1, _curvature_poly_order, false /*specific order only*/, V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                    for (int j=0; j<manifold_NP; ++j) {
                        P_target_row(offset, j) = t1(j);
                    }
                }
            });
        } else {
            compadre_kernel_assert_release((false) && "Functionality not yet available.");
        }
    }
}

KOKKOS_INLINE_FUNCTION
void GMLS::computeTargetFunctionalsOnManifold(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_right_type P_target_row, scratch_matrix_right_type V, scratch_matrix_right_type G_inv, scratch_vector_type curvature_coefficients, scratch_vector_type curvature_gradients) const {

    // only designed for 2D manifold embedded in 3D space
    const int target_index = _initial_index_for_batch + teamMember.league_rank();
    const int target_NP = this->getNP(_poly_order, _dimensions-1, _reconstruction_space);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, P_target_row.extent(0)), [&] (const int j) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, P_target_row.extent(1)),
          [=] (const int& k) {
            P_target_row(j,k) = 0;
        });
    });
    teamMember.team_barrier();

    // determine if additional evaluation sites are requested by user and handled by target operations 
    bool additional_evaluation_sites_need_handled = 
        (_additional_evaluation_coordinates.extent(0) > 0) ? true : false; // additional evaluation sites are specified

    const int num_evaluation_sites = (static_cast<int>(_additional_evaluation_indices.extent(1)) > 1) 
                ? static_cast<int>(getNAdditionalEvaluationCoordinates(target_index)+1) : 1;

    //auto blah = _target_extra_data;
    //if (target_index==1) {
    //    printf("TARGET INDEX 1!!!!!!!!!!!!!!\n");
    //    for (size_t j=0; j<blah.extent(1); ++j) {
    //        printf("TARGET VAL AT 1: %d: %f\n", j, blah(target_index,j));
    //    }
    //}
    //blah = _source_extra_data;
    //if (target_index==1) {
    //    printf("SOURCE INDEX 1!!!!!!!!!!!!!!\n");
    //    for (size_t j=0; j<blah.extent(1); ++j) {
    //        printf("SOURCE VAL AT 1: %d: %f\n", j, blah(target_index,j));
    //    }
    //}

    for (size_t i=0; i<_operations.size(); ++i) {

        bool additional_evaluation_sites_handled = false; // target operations that can handle these sites should flip this flag

        bool operation_handled = true;

        // USER defined targets on the manifold should be added to this file
        // if the USER defined targets don't catch this operation, then operation_handled will be false
        #include "USER_ManifoldTargetFunctionals.hpp"

        // if the user didn't handle the operation, we pass it along to the toolkit's targets
        if (!operation_handled) {
        if (_dimensions>2) {
            if (_operations(i) == TargetOperation::ScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    for (int j=0; j<num_evaluation_sites; ++j) { 
                        this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        for (int k=0; k<target_NP; ++k) {
                            P_target_row(offset, k) = t1(k);
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::VectorPointEvaluation) {
                // vector basis
                if (_reconstruction_space_rank == 1) {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                        for (int k=0; k<num_evaluation_sites; ++k) { 
                            // output component 0
                            this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, k);
                            int offset = getTargetOffsetIndexDevice(i, 0, 0, k);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = t1(j);
                                P_target_row(offset, target_NP + j) = 0;
                            }
                            offset = getTargetOffsetIndexDevice(i, 1, 0, k);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                                P_target_row(offset, target_NP + j) = 0;
                            }

                            // output component 1
                            offset = getTargetOffsetIndexDevice(i, 0, 1, k);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                                P_target_row(offset, target_NP + j) = 0;
                            }
                            offset = getTargetOffsetIndexDevice(i, 1, 1, k);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                                P_target_row(offset, target_NP + j) = t1(j);
                            }
                        }
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                // scalar basis times number of components in the vector
                } else if (_reconstruction_space_rank == 0) {
                    Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                        for (int k=0; k<num_evaluation_sites; ++k) { 
                            // output component 0
                            this->calcPij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, k);
                            int offset = getTargetOffsetIndexDevice(i, 0, 0, k);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = t1(j);
                            }
                            offset = getTargetOffsetIndexDevice(i, 1, 0, k);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }

                            // output component 1
                            offset = getTargetOffsetIndexDevice(i, 0, 1, k);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = 0;
                            }
                            offset = getTargetOffsetIndexDevice(i, 1, 1, k);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = t1(j);
                            }
                        }
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                }

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


                    const int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset, j) = 0;
                    }
                    // scaled
                    if (_poly_order > 0 && _curvature_poly_order > 1) {
                        P_target_row(offset, 1) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
                        P_target_row(offset, 2) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
                    }
                    if (_poly_order > 1 && _curvature_poly_order > 0) {
                        P_target_row(offset, 3) = (h*h+a2*a2)/den/(h*h);
                        P_target_row(offset, 4) = -2*a1*a2/den/(h*h);
                        P_target_row(offset, 5) = (h*h+a1*a1)/den/(h*h);
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
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                            P_target_row(offset, target_NP + j) = 0;
                        }
                        P_target_row(offset, 0) = c0a;
                        P_target_row(offset, 1) = c1a;
                        P_target_row(offset, 2) = c2a;
                        P_target_row(offset, target_NP + 0) = c0b;
                        P_target_row(offset, target_NP + 1) = c1b;
                        P_target_row(offset, target_NP + 2) = c2b;
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

                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                        }

                        // verified
                        if (_poly_order > 0 && _curvature_poly_order > 1) {
                            P_target_row(offset, 1) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
                            P_target_row(offset, 2) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
                        }
                        if (_poly_order > 1 && _curvature_poly_order > 0) {
                            P_target_row(offset, 3) = (h*h+a2*a2)/den/(h*h);
                            P_target_row(offset, 4) = -2*a1*a2/den/(h*h);
                            P_target_row(offset, 5) = (h*h+a1*a1)/den/(h*h);
                        }

                    });
                }
            } else if (_operations(i) == TargetOperation::VectorLaplacianPointEvaluation) {
                // vector basis
                if (_reconstruction_space_rank == 1) {
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
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = t1(j);
                            P_target_row(offset, target_NP + j) = 0;
                        }
                        offset = getTargetOffsetIndexDevice(i, 1, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                            P_target_row(offset, target_NP + j) = 0;
                        }

                        // output component 1
                        offset = getTargetOffsetIndexDevice(i, 0, 1, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                            P_target_row(offset, target_NP + j) = 0;
                        }
                        offset = getTargetOffsetIndexDevice(i, 1, 1, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                            P_target_row(offset, target_NP + j) = t1(j);
                        }

                    });
                // scalar basis times number of components in the vector
                } else if (_reconstruction_space_rank == 0) {
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
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = t1(j);
                        }
                        offset = getTargetOffsetIndexDevice(i, 1, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                        }

                        // output component 1
                        offset = getTargetOffsetIndexDevice(i, 0, 1, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                        }
                        offset = getTargetOffsetIndexDevice(i, 1, 1, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = t1(j);
                        }
                    });
                }
            } else if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                if (_reconstruction_space_rank == 0
                    && (_polynomial_sampling_functional == PointSample
                    || _polynomial_sampling_functional == ManifoldVectorPointSample)) {
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
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                        }
                        if (_poly_order > 0 && _curvature_poly_order > 0) {
                            P_target_row(offset, 1) = t1a + t2a;
                            P_target_row(offset, 2) = 0;
                        }

                        offset = getTargetOffsetIndexDevice(i, 0, 1, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                        }
                        if (_poly_order > 0 && _curvature_poly_order > 0) {
                            P_target_row(offset, 1) = 0;
                            P_target_row(offset, 2) = t1b + t2b;
                        }

                    });
                // staggered gradient w/ edge integrals performed by numerical integration
                // with a vector basis
                } else if (_reconstruction_space_rank == 1
                        && _polynomial_sampling_functional 
                            == StaggeredEdgeIntegralSample) {
                    compadre_kernel_assert_release((false) && "Functionality not yet available.");

                // staggered gradient w/ edge integrals known analytically, using a basis
                // of potentials
                } else if (_reconstruction_space_rank == 0
                        && _polynomial_sampling_functional 
                            == StaggeredEdgeAnalyticGradientIntegralSample) {
                    compadre_kernel_assert_release((false) && "Functionality not yet available.");

                } else {
                    compadre_kernel_assert_release((false) && "Functionality not yet available.");
                }
            } else if (_operations(i) == TargetOperation::DivergenceOfVectorPointEvaluation) {
                // vector basis
                if (_reconstruction_space_rank == 1
                        && _polynomial_sampling_functional == ManifoldVectorPointSample) {
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
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                            P_target_row(offset, target_NP + j) = 0;
                        }
                        P_target_row(offset, 0) = c0a;
                        P_target_row(offset, 1) = c1a;
                        P_target_row(offset, 2) = c2a;

                        // 2nd input component
                        offset = getTargetOffsetIndexDevice(i, 1, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                            P_target_row(offset, target_NP + j) = 0;
                        }
                        P_target_row(offset, target_NP + 0) = c0b;
                        P_target_row(offset, target_NP + 1) = c1b;
                        P_target_row(offset, target_NP + 2) = c2b;
                    });
                // scalar basis times number of components in the vector
                } else if (_reconstruction_space_rank == 0
                        && _polynomial_sampling_functional == ManifoldVectorPointSample) {
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
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                        }
                        P_target_row(offset, 0) = c0a;
                        P_target_row(offset, 1) = c1a;
                        P_target_row(offset, 2) = c2a;

                        // 2nd input component
                        offset = getTargetOffsetIndexDevice(i, 1, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                        }
                        P_target_row(offset, 0) = c0b;
                        P_target_row(offset, 1) = c1b;
                        P_target_row(offset, 2) = c2b;
                    });
                // staggered divergence acting on vector polynomial space
                } else if (_reconstruction_space_rank == 1
                        && _polynomial_sampling_functional 
                            == StaggeredEdgeIntegralSample) {

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
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = 0;
                            P_target_row(offset, target_NP + j) = 0;
                        }
                        P_target_row(offset, 0) = c0a;
                        P_target_row(offset, 1) = c1a;
                        P_target_row(offset, 2) = c2a;
                        P_target_row(offset, target_NP + 0) = c0b;
                        P_target_row(offset, target_NP + 1) = c1b;
                        P_target_row(offset, target_NP + 2) = c2b;

                    });
                } else {
                    compadre_kernel_assert_release((false) && "Functionality not yet available.");
                }
            } else if (_operations(i) == TargetOperation::GaussianCurvaturePointEvaluation) {
                double h = _epsilons(target_index);
                
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    for (int k=0; k<num_evaluation_sites; ++k) { 
                        XYZ relative_coord;
                        if (k > 0) {
                            for (int d=0; d<_dimensions-1; ++d) {
                                relative_coord[d] = getTargetAuxiliaryCoordinate(target_index, k, d, &V);
                                relative_coord[d] -= getTargetCoordinate(target_index, d, &V);
                            }
                        } else {
                            for (int i=0; i<3; ++i) relative_coord[i] = 0;
                        }

                        int offset = getTargetOffsetIndexDevice(i, 0, 0, k);
                        P_target_row(offset, 0) = GaussianCurvature(curvature_coefficients, h, relative_coord.x, relative_coord.y);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::CurlOfVectorPointEvaluation) {
                double h = _epsilons(target_index);
                int alphax, alphay;
                
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    for (int k=0; k<num_evaluation_sites; ++k) { 
                        XYZ relative_coord;
                        if (k > 0) {
                            for (int d=0; d<_dimensions-1; ++d) {
                                relative_coord[d] = getTargetAuxiliaryCoordinate(target_index, k, d, &V);
                                relative_coord[d] -= getTargetCoordinate(target_index, d, &V);
                            }
                        } else {
                            for (int i=0; i<3; ++i) relative_coord[i] = 0;
                        }

                        int offset = getTargetOffsetIndexDevice(i, 0, 0, k);
                        int index = 0;
                        for (int n = 0; n <= _poly_order; n++){
                            for (alphay = 0; alphay <= n; alphay++){
                                alphax = n - alphay;
                                P_target_row(offset, index) = SurfaceCurlOfScalar(curvature_coefficients, h, relative_coord.x, relative_coord.y, alphax, alphay, 0);
                                index++;
                            }
                        }

                        offset = getTargetOffsetIndexDevice(i, 0, 1, k);
                        index = 0;
                        for (int n = 0; n <= _poly_order; n++){
                            for (alphay = 0; alphay <= n; alphay++){
                                alphax = n - alphay;
                                P_target_row(offset, index) = SurfaceCurlOfScalar(curvature_coefficients, h, relative_coord.x, relative_coord.y, alphax, alphay, 1);
                                index++;
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else {
                compadre_kernel_assert_release((false) && "Functionality not yet available.");
            }
        } else if (_dimensions==2) { // 1D manifold in 2D problem
            if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset, j) = 0;
                        t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
                    }
                    for (int j=0; j<target_NP; ++j) {
                        double v1 = t1(j)*G_inv(0,0);
                        P_target_row(offset, j) = v1;
                    }
                });
            } else {
                compadre_kernel_assert_release((false) && "Functionality not yet available.");
            }
        }
        compadre_kernel_assert_release(((additional_evaluation_sites_need_handled && additional_evaluation_sites_handled) || (!additional_evaluation_sites_need_handled)) && "Auxiliary evaluation coordinates are specified by user, but are calling a target operation that can not handle evaluating additional sites.");
        } // !operation_handled
    }
}

}; // Compadre
#endif
