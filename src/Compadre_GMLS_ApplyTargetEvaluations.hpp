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
#ifndef _COMPADRE_GMLS_APPLY_TARGET_EVALUATIONS_HPP_
#define _COMPADRE_GMLS_APPLY_TARGET_EVALUATIONS_HPP_

#include "Compadre_GMLS.hpp"

namespace Compadre {

KOKKOS_INLINE_FUNCTION
void GMLS::applyTargetsToCoefficients(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, scratch_vector_type w, scratch_vector_type P_target_row, const int target_NP) const {

    const int target_index = teamMember.league_rank();

    if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) {
        // CPU
        for (int i=0; i<this->getNNeighbors(target_index); ++i) {
            for (int j=0; j<_operations.size(); ++j) {
                for (int k=0; k<_lro_output_tile_size[j]; ++k) {
                    for (int m=0; m<_lro_input_tile_size[j]; ++m) {
                        double alpha_ij = 0;
                        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
                            _basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
                            if (_sampling_multiplier>1 && m<_sampling_multiplier) {
                                talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l)*Q(ORDER_INDICES(i + m*this->getNNeighbors(target_index),l));
                            } else if (_sampling_multiplier == 1) {
                                talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l)*Q(ORDER_INDICES(i,l));
                            } else {
                                talpha_ij += 0;
                            }
                        }, alpha_ij);
                        Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                            _alphas(ORDER_INDICES(target_index, (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i)) = alpha_ij;
                        });
                    }
                }
            }
        }
    } else {
//        // GPU
//        for (int j=0; j<_operations.size(); ++j) {
//            for (int k=0; k<_lro_output_tile_size[j]; ++k) {
//                for (int m=0; m<_lro_input_tile_size[j]; ++m) {
//                    const int alpha_offset = (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0);
//                    const int P_offset =_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k);
//                    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,
//                        this->getNNeighbors(target_index)), [=] (const int i) {
//
//                        double alpha_ij = 0;
//                        if (_sampling_multiplier>1 && m<_sampling_multiplier) {
//                            const int m_neighbor_offset = i+m*this->getNNeighbors(target_index);
//                            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember,
//                                _basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
//                            //for (int l=0; l<_basis_multiplier*target_NP; ++l) {
//                                talpha_ij += P_target_row(P_offset + l)*Q(ORDER_INDICES(m_neighbor_offset,l));
//                            }, alpha_ij);
//                            //}
//                        } else if (_sampling_multiplier == 1) {
//                            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember,
//                                _basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
//                            //for (int l=0; l<_basis_multiplier*target_NP; ++l) {
//                                talpha_ij += P_target_row(P_offset + l)*Q(ORDER_INDICES(i,l));
//                            }, alpha_ij);
//                            //}
//                        } 
//                        Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
//                            t1(i) = alpha_ij;
//                        });
//                    });
//                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,
//                        this->getNNeighbors(target_index)), [=] (const int i) {
//                        _alphas(ORDER_INDICES(target_index, alpha_offset + i)) = t1(i);
//                    });
//                    teamMember.team_barrier();
//                }
//            }
//        }

        // GPU
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,
                this->getNNeighbors(target_index)), [=] (const int i) {
            for (int j=0; j<_operations.size(); ++j) {
                for (int k=0; k<_lro_output_tile_size[j]; ++k) {
                    for (int m=0; m<_lro_input_tile_size[j]; ++m) {
                        const int alpha_offset = (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0);
                        const int P_offset =_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k);

                            double alpha_ij = 0;
                            if (_sampling_multiplier>1 && m<_sampling_multiplier) {
                                const int m_neighbor_offset = i+m*this->getNNeighbors(target_index);
                                for (int l=0; l<_basis_multiplier*target_NP; ++l) {
                                    alpha_ij += P_target_row(P_offset + l)*Q(ORDER_INDICES(m_neighbor_offset,l));
                                }
                            } else if (_sampling_multiplier == 1) {
                                for (int l=0; l<_basis_multiplier*target_NP; ++l) {
                                    alpha_ij += P_target_row(P_offset + l)*Q(ORDER_INDICES(i,l));
                                }
                            } 
                            _alphas(ORDER_INDICES(target_index, alpha_offset + i)) = alpha_ij;
                    }
                }
            }
        });
    }
}

}; // Compadre
#endif
