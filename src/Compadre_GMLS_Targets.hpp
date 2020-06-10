#ifndef _COMPADRE_GMLS_TARGETS_HPP_
#define _COMPADRE_GMLS_TARGETS_HPP_

#include "Compadre_GMLS.hpp"
#include "Compadre_Manifold_Functions.hpp"

namespace Compadre {

KOKKOS_INLINE_FUNCTION
void GMLS::computeTargetFunctionals(const member_type& teamMember, scratch_vector_type delta, scratch_vector_type thread_workspace, scratch_matrix_right_type P_target_row) const {

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
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, delta.extent(0)), [&] (const int j) { delta(j) = 0; });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, thread_workspace.extent(0)), [&] (const int j) { thread_workspace(j) = 0; });
    teamMember.team_barrier();

    const int target_NP = this->getNP(_poly_order, _dimensions, _reconstruction_space);
    const int num_evaluation_sites = (static_cast<int>(_additional_evaluation_indices.extent(1)) > 1) 
                ? static_cast<int>(getNAdditionalEvaluationCoordinates(target_index)+1) : 1;

    for (size_t i=0; i<_operations.size(); ++i) {

        bool additional_evaluation_sites_handled = false; // target operations that can handle these sites should flip this flag

        bool operation_handled = true;

        // USER defined targets should be added to this file
        // if the USER defined targets don't catch this operation, then operation_handled will be false
        #include "Compadre_USER_StandardTargetFunctionals.hpp"

        // if the user didn't handle the operation, we pass it along to the toolkit's targets
        if (!operation_handled) {

        if (_reconstruction_space == ReconstructionSpace::ScalarTaylorPolynomial) {

            /*
             * Beginning of ScalarTaylorPolynomial basis
             */

            if (_operations(i) == TargetOperation::ScalarPointEvaluation || (_operations(i) == TargetOperation::VectorPointEvaluation && _dimensions == 1) /* vector is a scalar in 1D */) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                    for (int k=0; k<target_NP; ++k) {
                        P_target_row(offset, k) = delta(k);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::LaplacianOfScalarPointEvaluation) {
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
            } else if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    for (int d=0; d<_dimensions; ++d) {
                        int offset = getTargetOffsetIndexDevice(i, 0, d, j);
                        auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                        this->calcGradientPij(teamMember, row.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, d /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialXOfScalarPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                    auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                    this->calcGradientPij(teamMember, row.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialYOfScalarPointEvaluation) {
                compadre_kernel_assert_release(_dimensions>1 && "PartialYOfScalarPointEvaluation requested for dim < 2");
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                    auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                    this->calcGradientPij(teamMember, row.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialZOfScalarPointEvaluation) {
                compadre_kernel_assert_release(_dimensions>2 && "PartialZOfScalarPointEvaluation requested for dim < 3");
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                    auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                    this->calcGradientPij(teamMember, row.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
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
            } else if (_operations(i) == TargetOperation::ScalarFaceAverageEvaluation) {
                //printf("ScalarFaceAverageEvaluation\n");
                const double factorial[15] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};
                int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);

                // Calculate basis matrix for NON MANIFOLD problems
                double cutoff_p = _epsilons(target_index);
                int alphax, alphay, alphaz;
                double alphaf;

                double triangle_coords[9];
                scratch_matrix_right_type triangle_coords_matrix(triangle_coords, 3, 3); 
                scratch_vector_type midpoint(delta.data(), 3);

                getMidpointFromCellVertices(teamMember, midpoint, _target_extra_data, target_index, 3 /*dim*/);
                for (int j=0; j<3; ++j) {
                    triangle_coords_matrix(j, 0) = midpoint(j);
                }
                size_t num_vertices = _target_extra_data.extent(1) / 3;
                
                XYZ relative_coord;

                // loop over each two vertices 
                for (size_t v=0; v<num_vertices; ++v) {
                    int v1 = v;
                    int v2 = (v+1) % num_vertices;

                    for (int j=0; j<3; ++j) {
                        triangle_coords_matrix(j,1) = _target_extra_data(target_index, v1*3+j) - triangle_coords_matrix(j,0);
                        triangle_coords_matrix(j,2) = _target_extra_data(target_index, v2*3+j) - triangle_coords_matrix(j,0);
                    }
                    // triangle_coords now has:
                    // (midpoint_x, midpoint_y, midpoint_z, 
                    //  v1_x-midpoint_x, v1_y-midpoint_y, v1_z-midpoint_z, 
                    //  v2_x-midpoint_x, v2_y-midpoint_y, v2_z-midpoint_z);
                    auto T=triangle_coords_matrix;
                    for (int quadrature = 0; quadrature<_qm.getNumberOfQuadraturePoints(); ++quadrature) {
                        double transformed_qp[3] = {0,0,0};
                        for (int j=0; j<3; ++j) {
                            for (int k=1; k<3; ++k) {
                                transformed_qp[j] += T(j,k)*_qm.getSite(quadrature, k-1);
                            }
                            transformed_qp[j] += T(j,0);
                        }
                        // half the norm of the cross-product is the area of the triangle
                        // so scaling is area / reference area (0.5) = the norm of the cross-product
                        double area_scaling = getAreaFromVectors(teamMember, 
                                Kokkos::subview(T, Kokkos::ALL(), 1), Kokkos::subview(T, Kokkos::ALL(), 2));

                        for (int j=0; j<3; ++j) {
                            relative_coord[j] = transformed_qp[j] - getTargetCoordinate(target_index, j); // shift quadrature point by target site
                        }

                        int k = 0;

                        const int start_index = 0;
                        for (int n = start_index; n <= _poly_order; n++){
                            for (alphaz = 0; alphaz <= n; alphaz++){
                                int s = n - alphaz;
                                for (alphay = 0; alphay <= s; alphay++){
                                    alphax = s - alphay;
                                    alphaf = factorial[alphax]*factorial[alphay]*factorial[alphaz];
                                    double val_to_sum = (area_scaling * _qm.getWeight(quadrature) 
                                            * std::pow(relative_coord.x/cutoff_p,alphax)
                                            *std::pow(relative_coord.y/cutoff_p,alphay)
                                            *std::pow(relative_coord.z/cutoff_p,alphaz)/alphaf) / (0.5 * area_scaling);
                                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                                        if (quadrature==0) P_target_row(offset, k) = val_to_sum;
                                        else P_target_row(offset, k) += val_to_sum;
                                    });
                                    k++;
                                }
                            }
                        }
                    }
                }

            } else {
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
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                    for (int k=0; k<target_NP; ++k) {
                        P_target_row(offset, k) = delta(k);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::VectorPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                    this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, e);
                    for (int m=0; m<_sampling_multiplier; ++m) {
                        int output_components = _basis_multiplier;
                        for (int c=0; c<output_components; ++c) {
                            int offset = getTargetOffsetIndexDevice(i, m /*in*/, c /*out*/, e/*additional*/);
                            // for the case where _sampling_multiplier is > 1,
                            // this approach relies on c*target_NP being equivalent to P_target_row(offset, j) where offset is
                            // getTargetOffsetIndexDevice(i, m /*in*/, c /*out*/, e/*additional*/)*_basis_multiplier*target_NP;
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, c*target_NP + j) = delta(j);
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if ( (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) && (_polynomial_sampling_functional == StaggeredEdgeIntegralSample) ) {
                // when using staggered edge integral sample with vector basis, the gradient of scalar point evaluation
                // is just the vector point evaluation
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                    this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, e);
                    for (int m=0; m<_sampling_multiplier; ++m) {
                        int output_components = _basis_multiplier;
                        for (int c=0; c<output_components; ++c) {
                            int offset = getTargetOffsetIndexDevice(i, m /*in*/, c /*out*/, e/*additional*/);
                            // for the case where _sampling_multiplier is > 1,
                            // this approach relies on c*target_NP being equivalent to P_target_row(offset, j) where offset is
                            // getTargetOffsetIndexDevice(i, m /*in*/, c /*out*/, e/*additional*/)*_basis_multiplier*target_NP;
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, c*target_NP + j) = delta(j);
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                    int output_components = _basis_multiplier;
                    for (int m2=0; m2<output_components; ++m2) { // output components 2
                        int offset = getTargetOffsetIndexDevice(i, 0 /*in*/, m2 /*out*/, e);
                        this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, m2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, e);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = delta(j);
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::GradientOfVectorPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                    for (int m0=0; m0<_sampling_multiplier; ++m0) { // input components
                        int output_components = _basis_multiplier;
                        for (int m1=0; m1<output_components; ++m1) { // output components 1
                            for (int m2=0; m2<output_components; ++m2) { // output components 2
                                int offset = getTargetOffsetIndexDevice(i, m0 /*in*/, m1*output_components+m2 /*out*/, e);
                                this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, m2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, e);
                                for (int j=0; j<target_NP; ++j) {
                                    P_target_row(offset, m1*target_NP + j) = delta(j);
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
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                        for (int m=0; m<_sampling_multiplier; ++m) {
                            this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, m /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, e);
                            int offset = getTargetOffsetIndexDevice(i, m /*in*/, 0 /*out*/, e/*additional*/);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, m*target_NP + j) = delta(j);
                            }
                        }
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                }
            } else if (_operations(i) == TargetOperation::CurlOfVectorPointEvaluation) {
                if (_dimensions==3) { 
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
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
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
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
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                    for (int k=0; k<target_NP; ++k) {
                        P_target_row(offset, k) = delta(k);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::VectorPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                    this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, e);
                    for (int m=0; m<_sampling_multiplier; ++m) {
                        for (int c=0; c<_data_sampling_multiplier; ++c) {
                            int offset = getTargetOffsetIndexDevice(i, c /*in*/, c /*out*/, e/*additional*/);
                            for (int j=0; j<target_NP; ++j) {
                                P_target_row(offset, j) = delta(j);
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::LaplacianOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
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
            } else if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    for (int d=0; d<_dimensions; ++d) {
                        int offset = getTargetOffsetIndexDevice(i, 0, d, j);
                        auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                        this->calcGradientPij(teamMember, row.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, d /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::GradientOfVectorPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    for (int m0=0; m0<_dimensions; ++m0) { // input components
                        for (int m1=0; m1<_dimensions; ++m1) { // output components 1
                            for (int m2=0; m2<_dimensions; ++m2) { // output componets 2
                                int offset = getTargetOffsetIndexDevice(i, m0 /*in*/, m1*_dimensions+m2 /*out*/, j);
                                auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                                this->calcGradientPij(teamMember, row.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, m2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialXOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                    auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                    this->calcGradientPij(teamMember, row.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::PartialYOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
                if (_dimensions>1) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                        this->calcGradientPij(teamMember, row.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                }
            } else if (_operations(i) == TargetOperation::PartialZOfScalarPointEvaluation) {
                // copied from ScalarTaylorPolynomial
                if (_dimensions>2) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                        auto row = Kokkos::subview(P_target_row, offset, Kokkos::ALL());
                        this->calcGradientPij(teamMember, row.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                }
            } else if (_operations(i) == TargetOperation::DivergenceOfVectorPointEvaluation) {
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
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
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
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
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
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
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                    for (int m0=0; m0<_sampling_multiplier; ++m0) {
                        for (int m1=0; m1<_sampling_multiplier; ++m1) {

                          this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(m1+1) /* target is neighbor, but also which component */, 1 /*alpha*/, _dimensions, _poly_order, false /*bool on only specific order*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e /* evaluate at target */);

                          int offset = getTargetOffsetIndexDevice(i, m0 /*in*/, m1 /*out*/, e /*no additional*/);
                          for (int j=0; j<target_NP; ++j) {
                              P_target_row(offset, j) = delta(j);
                          }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::CurlOfVectorPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                    for (int m0=0; m0<_sampling_multiplier; ++m0) { // input components
                        for (int m1=0; m1<_sampling_multiplier; ++m1) { // output components
                            int offset = getTargetOffsetIndexDevice(i, m0 /*in*/, m1 /*out*/, e /*no additional*/);
                            if (_dimensions==3) {
                                switch (m1) {
                                    case 0:
                                        // output component 0
                                        this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(2+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u2y
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j)  = delta(j);
                                        }
                                        this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u1z
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        // u2y - u1z
                                        break;
                                    case 1:
                                        // output component 1
                                        this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(2+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u2x
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j)  = -delta(j);
                                        }
                                        this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u0z
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) += delta(j);
                                        }
                                        // -u2x + u0z
                                        break;
                                    default:
                                        // output component 2
                                        this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u1x
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j)  = delta(j);
                                        }
                                        this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u0y
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        // u1x - u0y
                                        break;
                                }
                            } else {
                                if (m1==0) {
                                    // curl results in 1D output
                                    P_target_row(offset, 2) = -std::pow(_epsilons(target_index), -1);
                                    P_target_row(offset, 3) = std::pow(_epsilons(target_index), -1);

                                    this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                    // u1x
                                    for (int j=0; j<target_NP; ++j) {
                                        P_target_row(offset, j)  = delta(j);
                                    }
                                    this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                    // -u0y
                                    for (int j=0; j<target_NP; ++j) {
                                        P_target_row(offset, j) -= delta(j);
                                    }
                                    // u1x - u0y
                                }
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::CurlCurlOfVectorPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                    for (int m0=0; m0<_sampling_multiplier; ++m0) { // input components
                        for (int m1=0; m1<_sampling_multiplier; ++m1) { // output components
                            int offset = getTargetOffsetIndexDevice(i, m0 /*in*/, m1 /*out*/, e /*no additional*/);
                            if (_dimensions == 3) {
                                switch (m1) {
                                    // manually compute the output components
                                    case 0:
                                        // output component 0
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction_1*/, 1 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u1xy
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j)  = delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction_1*/, 1 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u0yy
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(2+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction_1*/, 2 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u2xz
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) += delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 2 /*partial_direction_1*/, 2 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u0zz
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        // u1xy - u0yy + u2xz - u0zz
                                        break;
                                    case 1:
                                        // output component 1
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction_1*/, 0 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u1xx
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j)  = -delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction_1*/, 0 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u0yx
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) += delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(2+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction_1*/, 2 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u2yz
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) += delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 2 /*partial_direction_1*/, 2 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u1zz
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        // -u1xx + u0yx + u2yz - u1zz
                                        break;
                                    default:
                                        // output component 2
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(2+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction_1*/, 0 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u2xx
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j)  = -delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 2 /*partial_direction_1*/, 0 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u0zx
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) += delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(2+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction_1*/, 1 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u2yy
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 2 /*partial_direction_1*/, 1 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u1zy
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) += delta(j);
                                        }
                                        // -u2xx + u0zx - u2yy + u1zy
                                        break;
                                }
                            }
                            if (_dimensions == 2) {
                                // uses curl curl u = grad ( div (u) ) - laplace (u)
                                switch (m1) {
                                    case 0:
                                        // output component 0
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction_1*/, 0 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u0xx
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j)  = delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction_1*/, 0 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u1yx
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) += delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction_1*/, 0 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u0xx
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction_1*/, 1 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u0yy
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        // u0xx + u1yx - u0xx - u0yy
                                        break;
                                    case 1:
                                        // output component 1
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(0+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction_1*/, 1 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u0xy
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j)  = delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction_1*/, 1 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // u1yy
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) += delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 0 /*partial_direction_1*/, 0 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u1xx
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        this->calcHessianPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(1+1) /* -(component+1) */, 1 /*alpha*/, 1 /*partial_direction_1*/, 1 /*partial_direction_2*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                        // -u1yy
                                        for (int j=0; j<target_NP; ++j) {
                                            P_target_row(offset, j) -= delta(j);
                                        }
                                        // u0xy + u1yy - u1xx - u1yy
                                        break;
                                }
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::GradientOfVectorPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int e) {
                    for (int m0=0; m0<_sampling_multiplier; ++m0) { // input components
                        for (int m1=0; m1<_sampling_multiplier; ++m1) { // output components 1
                            for (int m2=0; m2<_sampling_multiplier; ++m2) { // output components 2
                                int offset = getTargetOffsetIndexDevice(i, m0 /*in*/, m1*_sampling_multiplier+m2 /*out*/, e);
                                this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, -(m1+1) /* target is neighbor */, 1 /*alpha*/, m2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial, VectorPointSample, e);
                                for (int j=0; j<target_NP; ++j) {
                                    P_target_row(offset, j) = delta(j);
                                }
                            }
                        }
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            }
        }  else {
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
void GMLS::computeCurvatureFunctionals(const member_type& teamMember, scratch_vector_type delta, scratch_vector_type thread_workspace, scratch_matrix_right_type P_target_row, const scratch_matrix_right_type* V, const local_index_type local_neighbor_index) const {

    compadre_kernel_assert_release(((int)thread_workspace.extent(0)>=(_curvature_poly_order+1)*_local_dimensions) && "Workspace thread_workspace not large enough.");

    const int target_index = _initial_index_for_batch + teamMember.league_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, P_target_row.extent(0)), [&] (const int j) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, P_target_row.extent(1)),
          [=] (const int& k) {
            P_target_row(j,k) = 0;
        });
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, delta.extent(0)), [&] (const int j) { delta(j) = 0; });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, thread_workspace.extent(0)), [&] (const int j) { thread_workspace(j) = 0; });
    teamMember.team_barrier();

    const int manifold_NP = this->getNP(_curvature_poly_order, _dimensions-1, ReconstructionSpace::ScalarTaylorPolynomial);
    for (size_t i=0; i<_curvature_support_operations.size(); ++i) {
        if (_curvature_support_operations(i) == TargetOperation::ScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, local_neighbor_index, 0 /*alpha*/, _dimensions-1, _curvature_poly_order, false /*bool on only specific order*/, V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                for (int j=0; j<manifold_NP; ++j) {
                    P_target_row(offset, j) = delta(j);
                }
            });
        } else if (_curvature_support_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
            Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                //int offset = i*manifold_NP;
                int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, local_neighbor_index, 0 /*alpha*/, 0 /*partial_direction*/, _dimensions-1, _curvature_poly_order, false /*specific order only*/, V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                for (int j=0; j<manifold_NP; ++j) {
                    P_target_row(offset, j) = delta(j);
                }
                if (_dimensions>2) { // _dimensions-1 > 1
                    //offset = (i+1)*manifold_NP;
                    offset = getTargetOffsetIndexDevice(i, 0, 1, 0);
                    this->calcGradientPij(teamMember, delta.data(), thread_workspace.data(), target_index, local_neighbor_index, 0 /*alpha*/, 1 /*partial_direction*/, _dimensions-1, _curvature_poly_order, false /*specific order only*/, V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                    for (int j=0; j<manifold_NP; ++j) {
                        P_target_row(offset, j) = delta(j);
                    }
                }
            });
        } else {
            compadre_kernel_assert_release((false) && "Functionality not yet available.");
        }
    }
}

KOKKOS_INLINE_FUNCTION
void GMLS::computeTargetFunctionalsOnManifold(const member_type& teamMember, scratch_vector_type delta, scratch_vector_type thread_workspace, scratch_matrix_right_type P_target_row, scratch_matrix_right_type V, scratch_matrix_right_type G_inv, scratch_vector_type curvature_coefficients, scratch_vector_type curvature_gradients) const {

    compadre_kernel_assert_release(((int)thread_workspace.extent(0)>=(_poly_order+1)*_local_dimensions) && "Workspace thread_workspace not large enough.");

    // only designed for 2D manifold embedded in 3D space
    const int target_index = _initial_index_for_batch + teamMember.league_rank();
    const int target_NP = this->getNP(_poly_order, _dimensions-1, _reconstruction_space);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, P_target_row.extent(0)), [&] (const int j) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember, P_target_row.extent(1)),
          [=] (const int& k) {
            P_target_row(j,k) = 0;
        });
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, delta.extent(0)), [&] (const int j) { delta(j) = 0; });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, thread_workspace.extent(0)), [&] (const int j) { thread_workspace(j) = 0; });
    teamMember.team_barrier();

    // determine if additional evaluation sites are requested by user and handled by target operations 
    bool additional_evaluation_sites_need_handled = 
        (_additional_evaluation_coordinates.extent(0) > 0) ? true : false; // additional evaluation sites are specified

    const int num_evaluation_sites = (static_cast<int>(_additional_evaluation_indices.extent(1)) > 1) 
                ? static_cast<int>(getNAdditionalEvaluationCoordinates(target_index)+1) : 1;

    for (size_t i=0; i<_operations.size(); ++i) {

        bool additional_evaluation_sites_handled = false; // target operations that can handle these sites should flip this flag

        bool operation_handled = true;

        // USER defined targets on the manifold should be added to this file
        // if the USER defined targets don't catch this operation, then operation_handled will be false
        #include "Compadre_USER_ManifoldTargetFunctionals.hpp"

        // if the user didn't handle the operation, we pass it along to the toolkit's targets
        if (!operation_handled) {
        if (_dimensions>2) {
            if (_operations(i) == TargetOperation::ScalarPointEvaluation) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int j) {
                    this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, j);
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, j);
                    for (int k=0; k<target_NP; ++k) {
                        P_target_row(offset, k) = delta(k);
                    }
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::VectorPointEvaluation) {
                // vector basis
                if (_reconstruction_space_rank == 1) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int k) {
                        // output component 0
                        this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, k);
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, k);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = delta(j);
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
                            P_target_row(offset, target_NP + j) = delta(j);
                        }
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                // scalar basis times number of components in the vector
                } else if (_reconstruction_space_rank == 0) {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int k) {
                        // output component 0
                        this->calcPij(teamMember, delta.data(), thread_workspace.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, ReconstructionSpace::ScalarTaylorPolynomial, PointSample, k);
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, k);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = delta(j);
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
                            P_target_row(offset, j) = delta(j);
                        }
                    });
                    additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
                }

            } else if (_operations(i) == TargetOperation::LaplacianOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

                    double h = _epsilons(target_index);
                    double a1=0, a2=0, a3=0, a4=0, a5=0;
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
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1=0, a2=0, a3=0, a4=0, a5=0;
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
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1=0, a2=0, a3=0, a4=0, a5=0;
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
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1=0, a2=0, a3=0, a4=0, a5=0;
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
                            delta(j) = 0;
                        }

                        // 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*Inv[G]*P
                        if (_poly_order > 0 && _curvature_poly_order > 1) {
                            delta(1) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4+(h*h+a1*a1)*a5))/den/den/(h*h);
                            delta(2) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4+(h*h+a1*a1)*a5))/den/den/(h*h);
                        }
                        if (_poly_order > 1 && _curvature_poly_order > 0) {
                            delta(3) = (h*h+a2*a2)/den/(h*h);
                            delta(4) = -2*a1*a2/den/(h*h);
                            delta(5) = (h*h+a1*a1)/den/(h*h);
                        }

                        // output component 0
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = delta(j);
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
                            P_target_row(offset, target_NP + j) = delta(j);
                        }

                    });
                // scalar basis times number of components in the vector
                } else if (_reconstruction_space_rank == 0) {
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1=0, a2=0, a3=0, a4=0, a5=0;
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
                            delta(j) = 0;
                        }

                        // 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*Inv[G]*P
                        if (_poly_order > 0 && _curvature_poly_order > 1) {
                            delta(1) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4+(h*h+a1*a1)*a5))/den/den/(h*h);
                            delta(2) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4+(h*h+a1*a1)*a5))/den/den/(h*h);
                        }
                        if (_poly_order > 1 && _curvature_poly_order > 0) {
                            delta(3) = (h*h+a2*a2)/den/(h*h);
                            delta(4) = -2*a1*a2/den/(h*h);
                            delta(5) = (h*h+a1*a1)/den/(h*h);
                        }

                        // output component 0
                        int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                        for (int j=0; j<target_NP; ++j) {
                            P_target_row(offset, j) = delta(j);
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
                            P_target_row(offset, j) = delta(j);
                        }
                    });
                }
            } else if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                if (_reconstruction_space_rank == 0
                    && (_polynomial_sampling_functional == PointSample
                    || _polynomial_sampling_functional == ManifoldVectorPointSample)) {
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

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
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1=0, a2=0, a3=0, a4=0, a5=0;
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
                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1=0, a2=0, a3=0, a4=0, a5=0;
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

                    Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

                        double h = _epsilons(target_index);
                        double a1=0, a2=0, a3=0, a4=0, a5=0;
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
                
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int k) {
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
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::CurlOfVectorPointEvaluation) {
                double h = _epsilons(target_index);
                
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_evaluation_sites), [=] (const int k) {
                    int alphax, alphay;
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
                });
                additional_evaluation_sites_handled = true; // additional non-target site evaluations handled
            } else if (_operations(i) == TargetOperation::ScalarFaceAverageEvaluation) {

                auto local_index = teamMember.league_rank();

                //printf("ScalarFaceAverageEvaluation\n");
                const double factorial[15] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};
                int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);


                // Calculate basis matrix for NON MANIFOLD problems
                double cutoff_p = _epsilons(target_index);
                int alphax, alphay;
                double alphaf;

                // global dimension cannot be determined in a constexpr way, so we use a largest case scenario
                // of dimensions 3 for _global_dimension
                double G_data[_global_dimensions*3];

                double triangle_coords[_global_dimensions*3];
                double triangle_coords_local[_local_dimensions*3];
                double triangle_coords_local_to_global[_global_dimensions*3];
                for (int i=0; i<_global_dimensions*3; ++i) G_data[i] = 0;
                for (int i=0; i<_global_dimensions*3; ++i) triangle_coords[i] = 0;
                for (int i=0; i<_local_dimensions*3; ++i) triangle_coords_local[i] = 0;
                for (int i=0; i<_global_dimensions*3; ++i) triangle_coords_local_to_global[i] = 0;
                // 3 is for # vertices in sub-triangle
                scratch_matrix_right_type G(G_data, _global_dimensions, 3); 
                scratch_matrix_right_type triangle_coords_matrix(triangle_coords, _global_dimensions, 3); 
                scratch_matrix_right_type triangle_coords_matrix_local(triangle_coords_local, _local_dimensions, 3); 
                scratch_matrix_right_type triangle_coords_matrix_local_to_global(triangle_coords_local_to_global, _global_dimensions, 3); 

                scratch_vector_type midpoint(delta.data(), _global_dimensions);
                //getMidpointFromCellVertices(teamMember, midpoint, _target_extra_data, target_index, _global_dimensions /*dim*/);
                for (int j=0; j<_global_dimensions; ++j) {
                    midpoint[j] = getTargetCoordinate(target_index, j);
                }
                for (int j=0; j<_global_dimensions; ++j) {
                    triangle_coords_matrix(j, 0) = midpoint(j);
                }
                if (_problem_type == ProblemType::MANIFOLD) {
                    // transform midpoint to local coords
                    XYZ midpoint_xyz;
                    for (int j=0; j<_global_dimensions; ++j) {
                        midpoint_xyz[j] = midpoint(j);
                    }
                    for (int j=0; j<_local_dimensions; ++j) {
                        triangle_coords_matrix_local(j, 0) = convertGlobalToLocalCoordinate(midpoint_xyz, j, &V);
                    }
                    XYZ midpoint_local;
                    for (int j=0; j<_local_dimensions; ++j) {
                        midpoint_local[j] = triangle_coords_matrix_local(j, 0);
                    }
                    for (int j=0; j<_global_dimensions; ++j) {
                        triangle_coords_matrix_local_to_global(j, 0) = convertLocalToGlobalCoordinate(midpoint_local, j, _local_dimensions, &V);
                    }
                }

                // NaN in last _global_dimensions indicates fewer vertices for this cell
                size_t num_vertices = (_target_extra_data(target_index, _target_extra_data.extent(1)-1)!=_target_extra_data(target_index, _target_extra_data.extent(1)-1)) ? _target_extra_data.extent(1) / _global_dimensions - 1 : _target_extra_data.extent(1) / _global_dimensions;
                double reference_cell_area = 0.5;
                double entire_cell_area = 0.0;
                auto T=triangle_coords_matrix;
                auto T_local = (_problem_type == ProblemType::MANIFOLD) ? triangle_coords_matrix_local : triangle_coords_matrix;
                //auto T_local = (_problem_type == ProblemType::MANIFOLD) ? triangle_coords_matrix : triangle_coords_matrix;
                //auto T_local = (_problem_type == ProblemType::MANIFOLD) ? triangle_coords_matrix_local_to_global : triangle_coords_matrix;
                //for (size_t v=0; v<num_vertices; ++v) {
                //    int v1 = v;
                //    int v2 = (v+1) % num_vertices;
                //    for (int j=0; j<_global_dimensions; ++j) {
                //        triangle_coords_matrix(j,1) = _target_extra_data(target_index, v1*_global_dimensions+j) - triangle_coords_matrix(j,0);
                //        triangle_coords_matrix(j,2) = _target_extra_data(target_index, v2*_global_dimensions+j) - triangle_coords_matrix(j,0);
                //    }
                //    entire_cell_area += 0.5 * getAreaFromVectors(teamMember, 
                //        Kokkos::subview(T, Kokkos::ALL(), 1), Kokkos::subview(T, Kokkos::ALL(), 2));
                //}

                const int max_num_rows = _sampling_multiplier*_max_num_neighbors;
                const int manifold_NP = this->getNP(_curvature_poly_order, _dimensions-1, ReconstructionSpace::ScalarTaylorPolynomial);
                const int max_manifold_NP = (manifold_NP > _NP) ? manifold_NP : _NP;
                const int this_num_cols = _basis_multiplier*max_manifold_NP;
                int P_dim_0, P_dim_1;
                getPDims(_dense_solver_type, _constraint_type, _reconstruction_space, _dimensions, max_num_rows, this_num_cols, P_dim_0, P_dim_1);
                int RHS_square_dim = getRHSSquareDim(_dense_solver_type, _constraint_type, _reconstruction_space, _dimensions, max_num_rows, this_num_cols);

    	        //scratch_matrix_right_type Q;
    	        //if (_dense_solver_type != DenseSolverType::LU) {
    	        //    // Solution from QR comes from RHS
    	        //    Q = scratch_matrix_right_type(_manifold_RHS.data()
    	        //        + TO_GLOBAL(local_index)*TO_GLOBAL(RHS_square_dim)*TO_GLOBAL(RHS_square_dim), RHS_square_dim, RHS_square_dim);
    	        //} else {
    	        //    // Solution from LU comes from P
    	        //    Q = scratch_matrix_right_type(_manifold_P.data()
    	        //        + TO_GLOBAL(local_index)*TO_GLOBAL(P_dim_1)*TO_GLOBAL(P_dim_0), P_dim_1, P_dim_0);
    	        //}
                std::vector<double> p_eval(manifold_NP,0);

                XYZ relative_coord;
                // loop over each two vertices 
                for (size_t v=0; v<num_vertices; ++v) {
                    int v1 = v;
                    int v2 = (v+1) % num_vertices;

                    for (int j=0; j<_global_dimensions; ++j) {
                        triangle_coords_matrix(j,1) = _target_extra_data(target_index, v1*_global_dimensions+j) - triangle_coords_matrix(j,0);
                        triangle_coords_matrix(j,2) = _target_extra_data(target_index, v2*_global_dimensions+j) - triangle_coords_matrix(j,0);
                    }
                    if (_problem_type == ProblemType::MANIFOLD) {
                        // transform midpoint to local coords
                        XYZ v1_xyz_global, v2_xyz_global, v1_xyz_local, v2_xyz_local, v1_xyz_local_to_global, v2_xyz_local_to_global;
                        for (int j=0; j<_global_dimensions; ++j) {
                            v1_xyz_global[j] = _target_extra_data(target_index, v1*_global_dimensions+j);
                            v2_xyz_global[j] = _target_extra_data(target_index, v2*_global_dimensions+j);
                        }
                        for (int j=0; j<_local_dimensions; ++j) {
                            v1_xyz_local[j] = convertGlobalToLocalCoordinate(v1_xyz_global, j, &V);
                            v2_xyz_local[j] = convertGlobalToLocalCoordinate(v2_xyz_global, j, &V);
                        }
                        for (int j=0; j<_local_dimensions; ++j) {
                            triangle_coords_matrix_local(j,1) = v1_xyz_local[j] - triangle_coords_matrix_local(j,0);
                            triangle_coords_matrix_local(j,2) = v2_xyz_local[j] - triangle_coords_matrix_local(j,0);
                        }
                        for (int j=0; j<_global_dimensions; ++j) {
                            v1_xyz_local_to_global[j] = convertLocalToGlobalCoordinate(v1_xyz_local, j, _local_dimensions, &V);
                            v2_xyz_local_to_global[j] = convertLocalToGlobalCoordinate(v2_xyz_local, j, _local_dimensions, &V);
                        }
                        for (int j=0; j<_global_dimensions; ++j) {
                            triangle_coords_matrix_local_to_global(j,1) = v1_xyz_local_to_global[j] - triangle_coords_matrix_local_to_global(j,0);
                            triangle_coords_matrix_local_to_global(j,2) = v2_xyz_local_to_global[j] - triangle_coords_matrix_local_to_global(j,0);
                        }
                    }
                    // triangle_coords now has:
                    // (midpoint_x, midpoint_y, midpoint_z, 
                    //  v1_x-midpoint_x, v1_y-midpoint_y, v1_z-midpoint_z, 
                    //  v2_x-midpoint_x, v2_y-midpoint_y, v2_z-midpoint_z);

                    for (int quadrature = 0; quadrature<_qm.getNumberOfQuadraturePoints(); ++quadrature) {
                        double transformed_qp[3] = {0,0,0};
                        for (int j=0; j<_global_dimensions; ++j) {
                            for (int k=1; k<3; ++k) { // 3 is for # vertices in subtriangle
                                transformed_qp[j] += T(j,k)*_qm.getSite(quadrature, k-1);
                            }
                            transformed_qp[j] += T(j,0);
                        }

                        double transformed_qp_norm = 0;
                        for (int j=0; j<_global_dimensions; ++j) {
                            transformed_qp_norm += transformed_qp[j]*transformed_qp[j];
                        }
                        transformed_qp_norm = std::sqrt(transformed_qp_norm);

                        // project back onto sphere
                        for (int j=0; j<_global_dimensions; ++j) {
                            transformed_qp[j] /= transformed_qp_norm;
                        }

                        //double transformed_qp[3] = {0,0,0};
                        //for (int j=0; j<_global_dimensions; ++j) {
                        //    for (int k=1; k<3; ++k) { // 3 is for # vertices in subtriangle
                        //        transformed_qp[j] += T(j,k)*_qm.getSite(quadrature, k-1);
                        //    }
                        //    transformed_qp[j] += T(j,0);
                        //}
                        // half the norm of the cross-product is the area of the triangle
                        // so scaling is area / reference area (0.5) = the norm of the cross-product
                        //double sub_cell_area = 0.5 * getAreaFromVectors(teamMember, 
                        //        Kokkos::subview(T, Kokkos::ALL(), 1), Kokkos::subview(T, Kokkos::ALL(), 2));//, V);
                        double scaling_factor = 1.0;//sub_cell_area / reference_cell_area;

                        XYZ qp = XYZ(transformed_qp[0], transformed_qp[1], transformed_qp[2]);
                        for (int j=0; j<2; ++j) {
                            XYZ qp = XYZ(transformed_qp[0], transformed_qp[1], transformed_qp[2]);
                            relative_coord[j] = convertGlobalToLocalCoordinate(qp,j,&V) - getTargetCoordinate(target_index,j,&V); // shift quadrature point by target site
                            //relative_coord[j] = transformed_qp[j] - getTargetCoordinate(target_index,j,&V); // shift quadrature point by target site
                            relative_coord[2] = 0;
                        }

                        //double grad_x1 = 0.0;
                        //double grad_x2 = 0.0;
                        double G_determinant = 1.0;
                        //if (_problem_type == ProblemType::MANIFOLD) {
                        //    std::fill(p_eval.begin(), p_eval.end(), 0);
                        //    int partial_direction = 0;
                        //    {
                        //        int alphax, alphay, alphaz;
                        //        double alphaf;
                        //        int m = 0;
                        //        for (int n = 0; n <= _curvature_poly_order; n++){
                        //            for (alphay = 0; alphay <= n; alphay++){
                        //                alphax = n - alphay;

                        //                int x_pow = (partial_direction == 0) ? alphax-1 : alphax;
                        //                int y_pow = (partial_direction == 1) ? alphay-1 : alphay;

                        //                if (x_pow<0 || y_pow<0) {
                        //                    p_eval[m] = 0;
                        //                } else {
                        //                    alphaf = factorial[x_pow]*factorial[y_pow];
                        //                    p_eval[m] = 1./cutoff_p 
                        //                                *std::pow(relative_coord.x/cutoff_p,x_pow)
                        //                                *std::pow(relative_coord.y/cutoff_p,y_pow)/alphaf;
                        //                }
                        //                m++;
                        //            }
                        //        }
                        //    }
                        //    for (int j=0; j<this->getNNeighbors(target_index); ++j) {
                        //        double alpha_ij = 0.0;
                        //        for (int k=0; k<manifold_NP; ++k) {
                        //            alpha_ij += p_eval[k]*Q(k,j);
                        //        }
                        //        XYZ rel_coord = getRelativeCoord(target_index, j, _global_dimensions, &V);
                        //        grad_x1 += alpha_ij * rel_coord[_global_dimensions-1];
                        //    }

                        //    std::fill(p_eval.begin(), p_eval.end(), 0);
                        //    partial_direction = 1;
                        //    {
                        //        int alphax, alphay, alphaz;
                        //        double alphaf;
                        //        int m = 0;
                        //        for (int n = 0; n <= _curvature_poly_order; n++){
                        //            for (alphay = 0; alphay <= n; alphay++){
                        //                alphax = n - alphay;

                        //                int x_pow = (partial_direction == 0) ? alphax-1 : alphax;
                        //                int y_pow = (partial_direction == 1) ? alphay-1 : alphay;

                        //                if (x_pow<0 || y_pow<0) {
                        //                    p_eval[m] = 0;
                        //                } else {
                        //                    alphaf = factorial[x_pow]*factorial[y_pow];
                        //                    p_eval[m] = 1./cutoff_p 
                        //                                *std::pow(relative_coord.x/cutoff_p,x_pow)
                        //                                *std::pow(relative_coord.y/cutoff_p,y_pow)/alphaf;
                        //                }
                        //                m++;
                        //            }
                        //        }
                        //    }
                        //    for (int j=0; j<this->getNNeighbors(target_index); ++j) {
                        //        double alpha_ij = 0.0;
                        //        for (int k=0; k<manifold_NP; ++k) {
                        //            alpha_ij += p_eval[k]*Q(k,j);
                        //        }
                        //        XYZ rel_coord = getRelativeCoord(target_index, j, _global_dimensions, &V);
                        //        grad_x2 += alpha_ij * rel_coord[_global_dimensions-1];
                        //    }

        		        //	// need to get 2x2 matrix of metric tensor
				        //	double G_0_0, G_0_1, G_1_0, G_1_1;
        		        //	G_0_0 = 1 + grad_x1*grad_x1;

        		        //	if (_local_dimensions>1) {
        		        //	    G_0_1 = grad_x1*grad_x2;
        		        //	    G_1_0 = grad_x2*grad_x1;
        		        //	    G_1_1 = 1 + grad_x2*grad_x2;
        		        //	}
                        //    G_determinant = std::sqrt(G_0_0*G_1_1 - G_0_1*G_1_0);
                        //    ////G_determinant = 1+grad_x1*grad_x1+grad_x2*grad_x2;
                        //    //double r1[_global_dimensions];
                        //    //double r2[_global_dimensions];
                        //    //scratch_vector_type kr1(r1, _global_dimensions); 
                        //    //scratch_vector_type kr2(r2, _global_dimensions); 
                        //    //for (int j=0; j<_global_dimensions; ++j) {
                        //    //    kr1[j]=V(0,j)+grad_x1*V(2,j);
                        //    //    kr2[j]=V(1,j)+grad_x2*V(2,j);
                        //    //}
                        //    //G_determinant = getAreaFromVectors(teamMember, kr1, kr2);

                        //    //double a1=0, a2=0, a3=0, a4=0, a5=0;
                        //    //if (_curvature_poly_order > 0) {
                        //    //    a1 = curvature_coefficients(1);
                        //    //    a2 = curvature_coefficients(2);
                        //    //}
                        //    //if (_curvature_poly_order > 1) {
                        //    //    a3 = curvature_coefficients(3);
                        //    //    a4 = curvature_coefficients(4);
                        //    //    a5 = curvature_coefficients(5);
                        //    //}
                        //    //double h = _epsilons(target_index);
                        //    //double alt_G_determinant = std::sqrt(1 + a1/pow(h,2) + a2/pow(h,2));
                        //    //printf("Gdet: %.16f vs %.16f\n", G_determinant, alt_G_determinant);
                        //}
                        //if (target_index==0) {
                        //    printf("quadrature: ( %f %f %f )\n", transformed_qp[0], transformed_qp[1], transformed_qp[2]);
                        //    printf("midpoint: ( %f %f %f )\n", midpoint[0], midpoint[1], midpoint[2]);
                        //    for (int j=0; j<_global_dimensions; ++j) {
                        //        printf("v1: %f\n", _target_extra_data(target_index, v1*_global_dimensions+j));
                        //        printf("v2: %f\n", _target_extra_data(target_index, v2*_global_dimensions+j));
                        //    }
                        //}
                        //for (int j=0; j<_global_dimensions; ++j) {
                        //    for (int k=0; k<_global_dimensions; ++k) {
                        //        G(j,k) = -0.5*2*transformed_qp[k]*transformed_qp[j]*std::pow(transformed_qp[0]*transformed_qp[0]+transformed_qp[1]*transformed_qp[1]+transformed_qp[2]*transformed_qp[2], -1.5) + (j==k)*std::pow(transformed_qp[0]*transformed_qp[0]+transformed_qp[1]*transformed_qp[1]+transformed_qp[2]*transformed_qp[2], -0.5);
                        //        //if (target_index==0) {
                        //        //    printf("G(%d,%d): %f\n", j, k, G(j,k));
                        //        //}
                        //    }
                        //}
                        //G_determinant = G(0,0)*(G(1,1)*G(2,2)-G(1,2)*G(2,1)) - G(0,1)*(G(1,0)*G(2,2)-G(2,0)*G(1,2)) + G(0,2)*(G(1,0)*G(2,1)-G(2,0)*G(1,1));
                        ////if (target_index==0) {
                        ////    printf("G det: %f\n", G_determinant);
                        ////}
                        ////G_determinant = std::sqrt(abs(G_determinant));
                        //G_determinant = 1./std::pow(std::sqrt(transformed_qp[0]*transformed_qp[0]+transformed_qp[1]*transformed_qp[1]+transformed_qp[2]*transformed_qp[2]),2);
                        double qp_norm_sq = transformed_qp_norm*transformed_qp_norm;
                        for (int j=0; j<_global_dimensions; ++j) {
                            G(j,1) = T(j,1)/transformed_qp_norm;
                            G(j,2) = T(j,2)/transformed_qp_norm;
                            for (int k=0; k<_global_dimensions; ++k) {
                                G(j,1) += transformed_qp[j]*(-0.5)*std::pow(qp_norm_sq,-1.5)*2*(transformed_qp[k]*T(k,1));
                                G(j,2) += transformed_qp[j]*(-0.5)*std::pow(qp_norm_sq,-1.5)*2*(transformed_qp[k]*T(k,2));
                            }
                            //for (int k=0; k<_global_dimensions; ++k) {
                            //    G(j,k) = 2*transformed_qp[j]*(-0.5*transformed_qp[k]*std::pow(transformed_qp[0]*transformed_qp[0]+transformed_qp[1]*transformed_qp[1]+transformed_qp[2]*transformed_qp[2], -1.5)) + (j==k)*std::pow(transformed_qp[0]*transformed_qp[0]+transformed_qp[1]*transformed_qp[1]+transformed_qp[2]*transformed_qp[2], -0.5);
                            //}
                        }
                        G_determinant = getAreaFromVectors(teamMember, 
                                Kokkos::subview(G, Kokkos::ALL(), 1), Kokkos::subview(G, Kokkos::ALL(), 2));//, *V);

                        int k = 0;
                        const int start_index = 0;
                        for (int n = start_index; n <= _poly_order; n++){
                            for (alphay = 0; alphay <= n; alphay++){
                                alphax = n - alphay;
                                alphaf = factorial[alphax]*factorial[alphay];
                                double val_to_sum = G_determinant * (scaling_factor * _qm.getWeight(quadrature) 
                                        * std::pow(relative_coord.x/cutoff_p,alphax)
                                        *std::pow(relative_coord.y/cutoff_p,alphay)/alphaf);// / entire_cell_area;
                                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                                    if (quadrature==0 && v==0) P_target_row(offset, k) = val_to_sum;
                                    else P_target_row(offset, k) += val_to_sum;
                                });
                                k++;
                            }
                        }
                        entire_cell_area += G_determinant * (scaling_factor * _qm.getWeight(quadrature));
                    }
                }
                int k = 0;
                for (int n = 0; n <= _poly_order; n++){
                    for (alphay = 0; alphay <= n; alphay++){
                        P_target_row(offset, k) /= entire_cell_area;
                        k++;
                    }
                }
            } else {
                compadre_kernel_assert_release((false) && "Functionality not yet available.");
            }
        } else if (_dimensions==2) { // 1D manifold in 2D problem
            if (_operations(i) == TargetOperation::GradientOfScalarPointEvaluation) {
                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                    int offset = getTargetOffsetIndexDevice(i, 0, 0, 0);
                    for (int j=0; j<target_NP; ++j) {
                        P_target_row(offset, j) = 0;
                        delta(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
                    }
                    for (int j=0; j<target_NP; ++j) {
                        double v1 = delta(j)*G_inv(0,0);
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

} // Compadre
#endif
