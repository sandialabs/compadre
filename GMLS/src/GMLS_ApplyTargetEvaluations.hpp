#ifndef _GMLS_APPLY_TARGET_EVALUATIONS_HPP_
#define _GMLS_APPLY_TARGET_EVALUATIONS_HPP_

#include "GMLS.hpp"

#ifdef COMPADRE_USE_KOKKOSCORE

KOKKOS_INLINE_FUNCTION
void GMLS::applySVD(const member_type& teamMember, scratch_matrix_type b_data, scratch_vector_type t1, scratch_matrix_type U, scratch_vector_type S, scratch_matrix_type V, scratch_vector_type w, scratch_matrix_type P_target_row, const int target_NP, const double abs_threshold) const {

	const int target_index = teamMember.league_rank();

	// t1 takes on the role of S inverse
	for (int i=0; i<target_NP*_basis_multiplier; i++) {
		t1(i) = ( std::abs(S(i)) > abs_threshold ) ? 1./S(i) : 0;
	}

	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) {
		// CPU
		for (int i=0; i<this->getNNeighbors(target_index); ++i) {

			for (int m=0; m<_sampling_multiplier; ++m) {
				for (int j=0; j<target_NP*_basis_multiplier; ++j) {
					double  bdataj = 0;
					Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
							target_NP*_basis_multiplier), [&] (const int k, double &tbdataj) {
		#if defined(USE_CUSTOM_SVD)
						tbdataj += V(j,k)*t1(k)*U(i + m*this->getNNeighbors(target_index),k);
		#else
						tbdataj += V(k,j)*t1(k)*U(i + m*this->getNNeighbors(target_index),k);
		#endif
					}, bdataj);
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
						b_data(j,m) = bdataj*std::sqrt(w(i + m*this->getNNeighbors(target_index)));
					});
					teamMember.team_barrier();
				}
			}
			teamMember.team_barrier();


			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {

						const int column = (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i;

						double alpha_ij = 0;
						Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
								_basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
							if (_sampling_multiplier>1 && m<_sampling_multiplier) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,m);
							} else if (_sampling_multiplier == 1) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,0);
							} else {
								talpha_ij += 0;
							}
						}, alpha_ij);
						Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
							// not optimal for LayoutLeft
							_alphas(target_index, column) = alpha_ij;
						});
						teamMember.team_barrier();
					}
				}
			}

			teamMember.team_barrier();
		}
	} else {
		// GPU
		for (int i=0; i<this->getNNeighbors(target_index); ++i) {
			for (int m=0; m<_sampling_multiplier; ++m) {
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,target_NP*_basis_multiplier), [=] (const int j) {
					double  bdataj = 0;
					for (int k=0; k<target_NP*_basis_multiplier; ++k) {
	#if defined(USE_CUSTOM_SVD)
						bdataj += V(j,k)*t1(k)*U(i + m*this->getNNeighbors(target_index),k);
	#else
						bdataj += V(k,j)*t1(k)*U(i + m*this->getNNeighbors(target_index),k);
	#endif
					}
					bdataj *= std::sqrt(w(i + m*this->getNNeighbors(target_index)));
					b_data(j,m) = bdataj;
				});
			}

			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {

						const int column = (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i;

						double alpha_ij = 0;
						Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
								_basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
							if (_sampling_multiplier>1 && m<_sampling_multiplier) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,m);
							} else if (_sampling_multiplier == 1) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,0);
							} else {
								talpha_ij += 0;
							}
						}, alpha_ij);
						Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
							// not optimal for LayoutLeft
							_alphas(target_index, column) = alpha_ij;
						});
						teamMember.team_barrier();
					}
				}
			}

			teamMember.team_barrier();
		}
	}
}

KOKKOS_INLINE_FUNCTION
void GMLS::applyTargetsToCoefficients(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, scratch_vector_type w, scratch_matrix_type P_target_row, const int target_NP) const {

	const int target_index = teamMember.league_rank();

	//GMLS_LinearAlgebra::upperTriangularBackSolve(teamMember, t1, t2, Q, R, w, _basis_multiplier*target_NP, _sampling_multiplier*this->getNNeighbors(target_index)); // stores solution in Q

	//if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
	//	// transpose Q
	//	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,_sampling_multiplier*this->getNNeighbors(target_index)), [=] (const int i) {
	//		Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,0,i), [&] (const int j) {
	//			double tmp = Q(i,j);
	//			Q(i,j) = Q(j,i);
	//			Q(j,i) = tmp;
	//		});
	//	});
	//}
	//teamMember.team_barrier();

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
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*Q(ORDER_INDICES(i + m*this->getNNeighbors(target_index),l));
							} else if (_sampling_multiplier == 1) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*Q(ORDER_INDICES(i,l));
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
//		// GPU
//		for (int j=0; j<_operations.size(); ++j) {
//			for (int k=0; k<_lro_output_tile_size[j]; ++k) {
//				for (int m=0; m<_lro_input_tile_size[j]; ++m) {
//					const int alpha_offset = (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0);
//					const int P_offset =_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k);
//					Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,
//						this->getNNeighbors(target_index)), [=] (const int i) {
//
//						double alpha_ij = 0;
//						if (_sampling_multiplier>1 && m<_sampling_multiplier) {
//							const int m_neighbor_offset = i+m*this->getNNeighbors(target_index);
//							Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember,
//								_basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
//							//for (int l=0; l<_basis_multiplier*target_NP; ++l) {
//								talpha_ij += P_target_row(P_offset + l, 0)*Q(ORDER_INDICES(m_neighbor_offset,l));
//							}, alpha_ij);
//							//}
//						} else if (_sampling_multiplier == 1) {
//							Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember,
//								_basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
//							//for (int l=0; l<_basis_multiplier*target_NP; ++l) {
//								talpha_ij += P_target_row(P_offset + l, 0)*Q(ORDER_INDICES(i,l));
//							}, alpha_ij);
//							//}
//						} 
//						Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
//							t1(i) = alpha_ij;
//						});
//					});
//					Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,
//						this->getNNeighbors(target_index)), [=] (const int i) {
//						_alphas(ORDER_INDICES(target_index, alpha_offset + i)) = t1(i);
//					});
//					teamMember.team_barrier();
//				}
//			}
//		}
//		// GPU
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
								alpha_ij += P_target_row(P_offset + l, 0)*Q(ORDER_INDICES(m_neighbor_offset,l));
							}
						} else if (_sampling_multiplier == 1) {
							for (int l=0; l<_basis_multiplier*target_NP; ++l) {
								alpha_ij += P_target_row(P_offset + l, 0)*Q(ORDER_INDICES(i,l));
							}
						} 
						_alphas(ORDER_INDICES(target_index, alpha_offset + i)) = alpha_ij;
				}
			}
		}
	});
	}
}

KOKKOS_INLINE_FUNCTION
void GMLS::applyMInverse(const member_type& teamMember, scratch_matrix_type b_data, scratch_matrix_type MInv, scratch_matrix_type PsqrtW, scratch_vector_type w, scratch_matrix_type P_target_row, const int target_NP) const {

	const int target_index = teamMember.league_rank();

	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) {
		// CPU
		for (int i=0; i<this->getNNeighbors(target_index); ++i) {
			for (int m=0; m<_sampling_multiplier; ++m) {
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,
					target_NP*_basis_multiplier), [=] (const int j) {
					double  bdataj = 0;
					for (int k=0; k<_NP; ++k) {
						bdataj += MInv(j,k)*PsqrtW(i,k);
					}
					bdataj *= std::sqrt(w(i + m*this->getNNeighbors(target_index)));
					b_data(j,m) = bdataj;
				});
			}
			teamMember.team_barrier();

			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {
						double alpha_ij = 0;
						Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
							_basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
							if (_sampling_multiplier>1 && m<_sampling_multiplier) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,m);
							} else if (_sampling_multiplier == 1) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,0);
							} 
						}, alpha_ij);
						Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
							_alphas(target_index, (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i) = alpha_ij;
						});
					}
				}
			}
		}
	} else {
		// GPU
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,
			this->getNNeighbors(target_index)), [=] (const int i) {
			for (int m=0; m<_sampling_multiplier; ++m) {
				for (int j=0; j<target_NP*_basis_multiplier; ++j) {
					double  bdataj = 0;
					for (int k=0; k<_NP; ++k) {
						bdataj += MInv(j,k)*PsqrtW(i,k);
					}
					bdataj *= std::sqrt(w(i + m*this->getNNeighbors(target_index)));
					b_data(j,m) = bdataj;
				}
			}

			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {
						double alpha_ij = 0;
						if (_sampling_multiplier>1 && m<_sampling_multiplier) {
							for (int l=0; l<_basis_multiplier*target_NP; ++l) {
								alpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,m);
							}
						} else if (_sampling_multiplier == 1) {
							for (int l=0; l<_basis_multiplier*target_NP; ++l) {
								alpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,0);
							}
						} 
						_alphas(target_index, (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i) = alpha_ij;
					}
				}
			}
		});
	}
}

#endif
#endif
