#ifndef _GMLS_LINEAR_ALGEBRA_DEFINITIONS_HPP_
#define _GMLS_LINEAR_ALGEBRA_DEFINITIONS_HPP_

#include "GMLS_LinearAlgebra_Declarations.hpp"
#define cudaStreamNonBlocking 0x01

namespace GMLS_LinearAlgebra {

KOKKOS_INLINE_FUNCTION
void upperTriangularBackSolve(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, scratch_vector_type w, int columns, int rows) {

	/*
	 * Backsolves R against rows of Q, scaling each row of Q by a factor of sqrt(W(that row))
	 * The solution is stored in Q
	 */

	const int target_index = teamMember.league_rank();

	// solving P^T * W * P * x = P^T * W * b, but first setting Pw = sqrt(W)*P
	// -> Pw^T * Pw * x= Pw^T * sqrt(W) * b
	// which is the normal equations for solving Pw * x = sqrt(W) * b
	// we factorized Pw as Pw = Q*R where Q is unitary
	// so Q*R*x= sqrt(W) * b
	// -> Q^T*Q*R*x=Q^T*sqrt(W)*b -> R*x=Q^T*sqrt(W)*b
	// so our right-hand sides are the basis for b are weighted rows of Q

	for (int i=columns-1; i>=0; --i) {
		const double r_i_i = R(i,i);
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, i, columns), [&] (const int j) {
			t2(j) = R(ORDER_INDICES(i,j)); // R is transposed
		});
		teamMember.team_barrier();

		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, rows), [&] (const int index) {

			t1(index) = Q(index,i) * std::sqrt(w(index));

			// this Q entry is known part of solution being subtracted from rhs
			// Q(index, i) is rhs, and Q(index, j) for j>i is known solution stored in Q
			for (int j=i+1; j<columns; ++j) {
				t1(index) -= Q(index,j) * t2(j);
			}

			Q(index,i) = t1(index) / r_i_i;
		});
		teamMember.team_barrier();
	}
}

KOKKOS_INLINE_FUNCTION
void GivensRotation(double& c, double& s, double a, double b) {
	double sign_of_a = (a < 0) ? -1 : 1;
	double sign_of_b = (b < 0) ? -1 : 1;

	c = 0; s = 0;
	if (b == 0) {
		c = sign_of_a;
		s = 0;
	} else if (a == 0) {
		c = 0;
		s = sign_of_b;
	} else if (std::abs(a) > std::abs(b)) {
		double t = b/a;
		double u = sign_of_a * std::abs(std::sqrt(1+t*t));
		c = 1./u;
		s = c*t;
	} else {
		double t = a/b;
		double u = sign_of_b * std::abs(std::sqrt(1+t*t));
		s = 1./u;
		c = s*t;
	}
}

KOKKOS_INLINE_FUNCTION
void createM(const member_type& teamMember, scratch_matrix_type M_data, scratch_matrix_type weighted_P, const int columns, const int rows) {
	/*
	 * Creates M = P^T * W * P
	 */

	const int target_index = teamMember.league_rank();

	for (int i=0; i<columns; ++i) {
		// offdiagonal entries
		for (int j=0; j<i; ++j) {
			double M_data_entry_i_j = 0;
			teamMember.team_barrier();

			Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int k, double &entry_val) {
				entry_val += weighted_P(ORDER_INDICES(k,i)) * weighted_P(ORDER_INDICES(k,j));
			}, M_data_entry_i_j );

			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
				M_data(i,j) = M_data_entry_i_j;
				M_data(j,i) = M_data_entry_i_j;
			});
			teamMember.team_barrier();
		}
		// diagonal entries
		double M_data_entry_i_j = 0;
		teamMember.team_barrier();

		Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int k, double &entry_val) {
			entry_val += weighted_P(ORDER_INDICES(k,i)) * weighted_P(ORDER_INDICES(k,i));
		}, M_data_entry_i_j );

		Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
			M_data(i,i) = M_data_entry_i_j;
		});
		teamMember.team_barrier();
	}
	teamMember.team_barrier();

//	for (int i=0; i<columns; ++i) {
//		for (int j=0; j<columns; ++j) {
//			std::cout << "(" << i << "," << j << "):" << M_data(i,j) << std::endl;
//		}
//	}
}

KOKKOS_INLINE_FUNCTION
void computeSVD(const member_type& teamMember, scratch_matrix_type U, scratch_vector_type S, scratch_matrix_type Vt, scratch_matrix_type P, int columns, int rows) {
#if not(defined(COMPADRE_USE_CUDA)) && defined(COMPADRE_USE_LAPACK) && defined(COMPADRE_USE_BOOST)

	Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

		const int target_index = teamMember.league_rank();

		/* Locals */
		int lda = rows,
			ldu = rows,
			ldvt = columns;

		int info, lwork;
		double wkopt;

		// LAPACK calls use Fortran data layout for matrices so we use boost to get smart pointer wrappers and avoid
		// manual indexing
		boost_matrix_fortran_type U_data_boost(rows, rows);
		boost_matrix_fortran_type Vt_data_boost(columns, columns);

		// temporary copy of data to put into Fortran layout
		boost_matrix_fortran_type P_data_boost(rows, columns);
		for(int k = 0; k < rows; k++){
			for(int l = 0; l < columns; l++){
				P_data_boost(k,l) = P(k,l);
			}
		}

		lwork = -1;
		// finds workspace size
		dgesvd_( (char *)"All", (char *)"All", &rows, &columns, &P_data_boost(0,0), &lda, (double *)S.data(), &U_data_boost(0,0), &ldu, &Vt_data_boost(0,0), &ldvt, &wkopt, &lwork, &info );


		lwork = (int)wkopt;
		const int clwork = lwork*sizeof(double);
		double work[clwork];

		// computes SVD
		dgesvd_( (char *)"All", (char *)"All", &rows, &columns, &P_data_boost(0,0), &lda, (double *)S.data(), &U_data_boost(0,0), &ldu, &Vt_data_boost(0,0), &ldvt, work, &lwork, &info );

		// check convergence
		if( info > 0 ) {
				printf( "The algorithm computing SVD failed to converge.\n" );
				exit( 1 );
		}

		// temporary copy back to kokkos type
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < rows; j++){
				U(i,j) = U_data_boost(i,j);
			}
		}
		for(int i = 0; i < columns; i++){
			for(int j = 0; j < columns; j++){
				Vt(i,j) = Vt_data_boost(i,j);
			}
		}

	});
#else
	//printf( "Either SVD requested on a GPU, or SVD requested on a CPU without LAPACK. Not implemented.\n" );
	//exit( 1 );
#endif
}

KOKKOS_INLINE_FUNCTION
void invertM(const member_type& teamMember, scratch_vector_type y, scratch_matrix_type M_inv, scratch_matrix_type L, scratch_matrix_type M_data, const int columns) {

	const int target_index = teamMember.league_rank();

#ifdef COMPADRE_USE_CUDA

//	// Construct L for A=L*L^T
//	for (int i=0; i<columns; ++i) {
//
//		for (int j=0; j<i; ++j) {
//			double sum = 0;
//			teamMember.team_barrier();
//			if (i>0) {
//				Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,0,i-1), [=] (const int index, double &temp_sum) {
//					temp_sum += L(j,index) * L(i,index);
//				}, sum);
//			} else {
//				sum = 0;
//			}
//			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
//				L(i,j) = 1./L(j,j) * (M_data(i,j) - sum);
//			});
//			std::cout << i << " " << j << " " << L(i,j) << std::endl;
//			teamMember.team_barrier();
//		}
//		double sum = 0;
//		if (i>0) {
//			Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,0,i-1), [=] (const int index, double &temp_sum) {
//				temp_sum += L(i,index) * L(i,index);
//			}, sum);
//		} else {
//			sum = 0;
//		}
//		Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
//			L(i,i) = std::sqrt(M_data(i,i) - sum);
//			std::cout << M_data(i,i) << " " << L(i,i) << std::endl;
//		});
//		teamMember.team_barrier();
//
//	}
//
////	teamMember.team_barrier();
////	 build identity matrix
//	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int i) {
//		for(int j = 0; j < columns; ++j) {
//			M_inv(i,j) = (i==j) ? 1 : 0;
//		}
//	});
//
//	// solve L*L^T M_inv = identity(columns,columns);
//	// L*(Y) = identity(columns,columns);, Y=L^T * M_inv
//	// L^T * M_inv = Y
//	// requires two back solves of a triangular matrix
//
//	// loop over columns of the identity matrix
//	teamMember.team_barrier();
//	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int index) {
//		// back-substitute to solve Ly = column_index(I)
//		for (int i=0; i<columns; i++) {
//			for (int j=0; j<=i; j++) {
//				y(i) = 0;
//				if (i==j) {
//					y(i) = M_inv(i, index) / L(i,i);
//				} else {
//					M_inv(i, index) -= L(i,j) * y(j);
//				}
//			}
//		}
//
////		for (int i=columns-1; i>=index; --i) { // SYMMETRY
//		for (int i=columns-1; i>=0; --i) {
//			for (int j=columns-1; j>=i; --j) {
//				if (i==j) {
//					M_inv(i, index) = y(i) / L(i,i);
//				} else {
//					y(i) -= L(j,i) * M_inv(j, index);
//				}
//			}
//		}
//	});
//	teamMember.team_barrier();

#elif defined(COMPADRE_USE_BOOST)

	Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
	//	 temporary copy of data to stay compatible with boost
		boost::numeric::ublas::identity_matrix<double> identity_matrix(columns,columns);
		boost_matrix_type M_inv_boost;
		M_inv_boost = boost_matrix_type(columns, columns);
		M_inv_boost.assign(identity_matrix);

		// temporary copy of data to stay compatible with boost
		boost_matrix_type M_data_boost(columns, columns, 0);
		for(int k = 0; k < columns; k++){
			for(int l = 0; l < columns; l++){
				M_data_boost(k,l) = M_data(k,l);
			}
		}

		// boost inversion
		boost::numeric::ublas::permutation_matrix<boost_matrix_type::size_type> p(columns);
		int attempt = boost::numeric::ublas::lu_factorize(M_data_boost, p);
		boost::numeric::ublas::lu_substitute(M_data_boost, p, M_inv_boost);

		// temporary copy back to kokkos type
		for(int i = 0; i < columns; i++){
			for(int j = 0; j < columns; j++){
				M_inv(i,j) = M_inv_boost(i,j);
			}
		}
	});

#endif
}

KOKKOS_INLINE_FUNCTION
void constructQ(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, scratch_vector_type tau, int columns, int rows) {

	const int target_index = teamMember.league_rank();

	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [&] (const int j) {
		Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,rows), [&] (const int i) {
			Q(i,j) = (i==j) ? 1 : 0;
		});
	});

	for (int j = 0; j<columns; j++) {
		double tau_j = tau(j);
		Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,j,rows), [&] (const int k) {
			t1(k) = (k==j) ? 1.0 : R(k,j);
		});
		teamMember.team_barrier();
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [&] (const int k) {
		// outer product of Q*w and tau*w

			double t2k = 0;
			Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember,j, rows), [&] (const int l, double& t_t2k) {
				t_t2k += Q(ORDER_INDICES(k,l))*t1(l);
				//t_t2k += Q(k,l)*t1(l);
			}, t2k);
			Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,j, rows), [&] (const int l) {
				Q(ORDER_INDICES(k,l)) -= t2k*tau_j*t1(l);
				//Q(k,l) -= t2k*tau_j*t1(l);
			});
		});
		teamMember.team_barrier();
	}
	teamMember.team_barrier();



	// transpose Q
	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [&] (const int i) {
		Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,0,i), [&] (const int j) {
		//Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
			//for(int j = 0; j < i; ++j) {
				double tmp = Q(i,j);
				Q(i,j) = Q(j,i);
				Q(j,i) = tmp;
			//}
		});
	});
	teamMember.team_barrier();
}

KOKKOS_INLINE_FUNCTION
void GivensQR(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_vector_type t3, scratch_matrix_type Q, scratch_matrix_type R, int columns, int rows) {

//#if not(defined(COMPADRE_USE_CUDA)) && defined(COMPADRE_USE_LAPACK)
//	// Works, but not thread safe with MPI + threads calling LAPACK
//
//	const int target_index = teamMember.league_rank();
//
//	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
//		for(int j = 0; j < rows; ++j) {
//			Q(ORDER_INDICES(i,j)) = (i==j) ? 1 : 0;
//		}
//	});
//	teamMember.team_barrier();
//
//	Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
//
//		/* Locals */
//		int lda = rows;//R.dimension_0();
//		int info, lwork;
//		double wkopt;
//
//		// temporary copy of data to put into Fortran layout
//		//boost_matrix_fortran_type P_data_boost(R.dimension_0(), R.dimension_1());
//		boost_matrix_fortran_type P_data_boost(rows, columns);
//		for(int k = 0; k < rows; k++){
//		      for(int l = 0; l < columns; l++){
//		              P_data_boost(k,l) = R(k,l);
//		      }
//		}
//
//		// LAPACK calls use Fortran data layout so we need LayoutLeft but have LayoutRight
//		//GMLS_LinearAlgebra::matrixToLayoutLeft(teamMember, t1, t2, R, columns, rows);
//
//		// indicates seeking optimal workspace size for performing QR decomp
//		// not necessary if we already know this, which we do in the size of t3
//		//lwork = -1;
//
//		//// finds workspace size
//		//dgeqrf_(&rows, &columns, (double *)R.data(), &lda,
//		//        (double *)t1.data(), &wkopt, &lwork, &info);
//
//
//		// allocates space needed to perform QR
//		lwork = static_cast<int>(t3.dimension_0());//(int)wkopt;
//		//lwork = (int)wkopt;
//		//printf("%d vs %d\n", static_cast<int>(t3.dimension_0()), lwork);
//
//		// computes QR
//		//dgeqrf_(&rows, &columns, (double *)R.data(), &lda,
//		//        (double *)t1.data(), (double *)t3.data(), &lwork, &info);
//		dgeqrf_(&rows, &columns, (double *)&P_data_boost(0,0), &lda,
//		        (double *)t1.data(), (double *)t3.data(), &lwork, &info);
//
//		//// indicates seeking optimal workspace size for reconstructing Q
//		//// not necessary if we already know this, which we do in the size of t3
//		//lwork = -1;
//
//		//if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) { 
//		//	// transposes Q since the Q expected would be LayoutLeft
//		//	dormqr_( (char *)"L", (char *)"T", &rows, &rows, 
//                //	         &columns, (double *)R.data(), &lda, (double *)t1.data(), 
//                //	         (double *)Q.data(), &rows, &wkopt, &lwork, &info);
//		//} else {
//		//	dormqr_( (char *)"L", (char *)"N", &rows, &rows, 
//                //	         &columns, (double *)R.data(), &lda, (double *)t1.data(), 
//                //	         (double *)Q.data(), &rows, &wkopt, &lwork, &info);
//		//}
//
//		lwork = static_cast<int>(t3.dimension_0());//(int)wkopt;
//		//lwork = (int)wkopt;
//		//printf("%d vs %d\n", static_cast<int>(t3.dimension_0()), lwork);
//
//		if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) { 
//			// transposes Q since the Q expected would be LayoutLeft
//			//dormqr_( (char *)"L", (char *)"T", &rows, &rows, 
//                	//         &columns, (double *)R.data(), &lda, (double *)t1.data(), 
//                	//         (double *)Q.data(), &rows, (double *)t3.data(), &lwork, &info);
//			dormqr_( (char *)"L", (char *)"T", &rows, &rows, 
//                	         &columns, (double *)&P_data_boost(0,0), &lda, (double *)t1.data(), 
//                	         (double *)Q.data(), &rows, (double *)t3.data(), &lwork, &info);
//		} else {
//			dormqr_( (char *)"L", (char *)"N", &rows, &rows, 
//                	         &columns, (double *)R.data(), &lda, (double *)t1.data(), 
//                	         (double *)Q.data(), &rows, (double *)t3.data(), &lwork, &info);
//		}
//
//		// check convergence
//		if( info > 0 ) {
//				printf( "The algorithm computing QR failed to converge.\n" );
//				exit( 1 );
//		}
//		for(int k = 0; k < rows; k++){
//		      for(int l = 0; l < columns; l++){
//		              R(k,l) = P_data_boost(k,l);
//		      }
//		}
//
//		// Put R back into LayoutRight
//		//GMLS_LinearAlgebra::matrixToLayoutLeft(teamMember, t1, t2, R, columns, rows);
//
//	});
//	teamMember.team_barrier();
//#else

	/*
	 * Performs a QR and overwrites the input matrix with the R matrix
	 * Stores and computes a full Q matrix
	 */

	const int target_index = teamMember.league_rank();

//	Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//		// checks for any nonzeros off of diagonal and superdiaganal entries
//		for (int k=0; k<m; k++) {
//			for (int l=0; l<n; l++) {
//				if (R(k,l)!=0 && target_index==0)
//				printf("%i %i %0.16f\n", k, l, R(k,l) );// << " " << l << " " << R(k,l) << std::endl;
//			}
//		}
//	});

	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
		for(int j = 0; j < rows; ++j) {
			Q(ORDER_INDICES(j,i)) = (i==j) ? 1 : 0;
		}
	});
	teamMember.team_barrier();

	double c, s;

	// nonoptimal for either layout of R to get the first row/column
	// this strategy with t2 is for performance and avoiding cache misses

	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int l) {
		t2(l) = R(ORDER_INDICES(l,0));
	});

	for (int j=0; j<columns; ++j) {

		teamMember.team_barrier();

		// Read in needed values from R which are stored in t2 either from 
		// loading them in for the first row/col or from gathering them in
		// t2 while performing optimal access computations
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j,rows), [=] (const int l) {
			t1(l) = t2(l);
		});

		for (int k=rows-2; k>=j; --k) {

			teamMember.team_barrier();

			// needs access to R(k,j), R(k+1,j) from original R matrix
			GMLS_LinearAlgebra::GivensRotation(c, s, t1(k), t1(k+1));

			teamMember.team_barrier();

			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int l) {
				double tmp_val_1 = Q(ORDER_INDICES(k,l));
				double tmp_val_2 = Q(ORDER_INDICES(k+1,l));
				Q(ORDER_INDICES(k,l)) = tmp_val_1*c + tmp_val_2*s;
				Q(ORDER_INDICES(k+1,l)) = tmp_val_1*-s + tmp_val_2*c;
			});

			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j,columns), [=] (const int l) {
				double tmp_val_1 = R(ORDER_INDICES(k,l));
				double tmp_val_2 = R(ORDER_INDICES(k+1,l));
				R(ORDER_INDICES(k,l)) = c*tmp_val_1 + s*tmp_val_2;
				if (l==j) t1(k) = R(ORDER_INDICES(k,l));

				R(ORDER_INDICES(k+1,l)) = -s*tmp_val_1 + c*tmp_val_2;
				// this is the last time that R(k,j) needed in the next column loop is changed
				if (l==j+1) t2(k+1) = R(ORDER_INDICES(k+1,l));
			});
		}
	}
	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) {
		// transpose Q
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
			for(int j = 0; j < i; ++j) {
				double tmp = Q(i,j);
				Q(i,j) = Q(j,i);
				Q(j,i) = tmp;
			}
		});
	}
	//Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
	//	// checks for any nonzeros off of diagonal and superdiagonal entries
	//	for (int k=0; k<rows; k++) {
	//		for (int l=0; l<columns; l++) {
	//			if (R(k,l)!=0 && target_index==0)
	//			printf("%i %i %0.16f\n", k, l, R(k,l) );// << " " << l << " " << R(k,l) << std::endl;
	//		}
	//	}
	//});
//#endif

}

KOKKOS_INLINE_FUNCTION
void HouseholderQR(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, scratch_vector_type tau, const int columns, const int rows) {
//	/*
//	 * Performs a QR and overwrites the input matrix with the R matrix
//	 * Stores and computes a full Q matrix
//	 *
//	 * Only written with LayoutLeft in mind, and even then GivensQR is faster
//	 */
//
//	const int target_index = teamMember.league_rank();
//
//	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
//		for(int j = 0; j < rows; ++j) {
//			Q(ORDER_INDICES(i,j)) = (i==j) ? 1 : 0;
//		}
//	});
//	teamMember.team_barrier();
//
//	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
//		// transpose R
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
//			for(int j = 0; j < i; ++j) {
//				double tmp = R(i,j);
//				R(i,j) = R(j,i);
//				R(j,i) = tmp;
//			}
//		});
//	}
//
//	for (int j = 0; j<columns; j++) {
//
//		teamMember.team_barrier();
//
//		double normx = 0;
//		Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,j,rows), [&] (const int i, double &temp_normx) {
//			temp_normx += R(i,j)*R(i,j);
//		}, normx);
//		teamMember.team_barrier();
//		normx = std::sqrt(normx);
//
//		double s = (R(j,j) >= 0) ? -1 : 1; // flips sign R(j,j)
//		double u1 = R(j,j) - s*normx;
//
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j+1,rows), [&] (const int i) {
//			t1(i) = R(i,j) / u1;
//		});
//		t1(j) = 1;
//
//		double tau = -s*u1/normx;
//
//		teamMember.team_barrier();
//
//		// outer product of tau*w and w'*R(j:end,:)
//		// t1 is w, so use it to get w'*R(j:end,:)
//		// t2 is 1 by n
//		for (int k=j; k<columns; k++) {
//			double t2k = 0;
//			Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,j,rows), [=] (const int l, double &temp_t2k) {
//				temp_t2k += t1(l)*R(l,k);
//			}, t2k);
//			Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//				t2(k) = t2k;
//			});
//		}
//		teamMember.team_barrier();
//
//
//		// only the diagonal of column j is affected
//		// the remainder is left intact (even though it is effectively zeroed out)
//		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//			R(j,j) -= tau*t1(j)*t2(j);
//			t2(j) = tau; // t2 carries all tau's when completed
//			// t1(j) is never changed again so t1 carries the first entry of each projector vector
//		});
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j+1,rows), [=] (const int k) {
//			R(k,j) = t1(k); // columns of R below the diagonal carry the remainder of each projector vector
//		});
//		teamMember.team_barrier();
//		// at this point, we have access to all parts of the projectors in R below the diagonal,
//		// and in t1, with taus in t2
//
//
//		// apply projector to remainder of R
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j,rows), [=] (const int k) {
//			for (int l=j+1; l<columns; l++) {
//				R(k,l) -= tau*t1(k)*t2(l);
//			}
//		});
//		teamMember.team_barrier();
//	}
//
//	// demonstrates that R below diagonal, t1, and t2 contain all projector data along with taus
//	// and it works
//	for (int j = 0; j<columns; j++) {
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int k) {
//		// reuse t2 to be n by 1
//		// outer product of Q*w and tau*w
//
//			double t2k = Q(k,j)*t1(j);
//			for (int l=j+1; l<rows; l++) {
//				t2k += Q(k,l)*R(l,j);
//			}
//
//			Q(k,j) -= t2k*t2(j)*t1(j);
//			for (int l=j+1; l<rows; l++) {
//				Q(k,l) -= t2k*t2(j)*R(l,j);
//			}
//		});
//	}
//
//	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
//		// transpose R
//		teamMember.team_barrier();
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
//			for(int j = 0; j < i; ++j) {
//				double tmp = R(i,j);
//				R(i,j) = R(j,i);
//				R(j,i) = tmp;
//			}
//		});
//	}

	/*
	 * Performs a QR and overwrites the input matrix with the R matrix
	 * Stores and computes a full Q matrix
	 *
	 * Only written with LayoutLeft in mind, and even then GivensQR is faster
	 */

	const int target_index = teamMember.league_rank();

//	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
//		for(int j = 0; j < rows; ++j) {
//			Q(ORDER_INDICES(i,j)) = (i==j) ? 1 : 0;
//		}
//	});
//	teamMember.team_barrier();

//	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
//		// transpose R
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
//			Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,0,i), [=] (const int j) {
//				double tmp = R(i,j);
//				R(i,j) = R(j,i);
//				R(j,i) = tmp;
//			});
//		});
//	}

	for (int j = 0; j<columns; j++) {

		teamMember.team_barrier();

		double normx = 0;
		Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember,j,rows), [=] (const int i, double &temp_normx) {
			temp_normx += R(i,j)*R(i,j);
		}, normx);
		teamMember.team_barrier();
		normx = std::sqrt(normx);

		double s = (R(j,j) >= 0) ? -1 : 1; // flips sign R(j,j)
		double u1 = R(j,j) - s*normx;

		Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,j,rows), [=] (const int i) {
			t1(i) = (i==j) ? 1.0 : R(i,j) / u1;
		});

		double tau = -s*u1/normx;

		teamMember.team_barrier();

		// outer product of tau*w and w'*R(j:end,:)
		// t1 is w, so use it to get w'*R(j:end,:)
		// t2 is 1 by n
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j,columns), [=] (const int k) {
		//for (int k=j; k<columns; k++) {
			//Kokkos::single(Kokkos::PerThread(teamMember), [=] () {
		
			//double t2k = 0;
			//for (int l=j; l<rows; ++l) t2k += t1(l)*R(l,k);
			//t2(k) = t2k;
			double t2k = 0;
			Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(teamMember,j,rows), [=] (const int l, double &temp_t2k) {
				temp_t2k += t1(l)*R(l,k);
			}, t2k);
			////t2(k) = t2k;

			//// TEMPORARY, RIGHT HERE. Seems to be too much work in loops

			Kokkos::single(Kokkos::PerThread(teamMember), [=] () {
			      t2(k) = t2k;
			});
		});
		teamMember.team_barrier();


		// only the diagonal of column j is affected
		// the remainder is left intact (even though it is effectively zeroed out)
		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
			Kokkos::single(Kokkos::PerThread(teamMember), [=] () {
				R(j,j) -= tau*t1(j)*t2(j);
				t2(j) = tau; // t2 carries all tau's when completed
				// t1(j) is never changed again so t1 carries the first entry of each projector vector
			});
		});
		Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,j+1,rows), [=] (const int k) {
			R(k,j) = t1(k); // columns of R below the diagonal carry the remainder of each projector vector
		});
		teamMember.team_barrier();
		// at this point, we have access to all parts of the projectors in R below the diagonal,
		// and in t1, with taus in t2


		// apply projector to remainder of R
		//Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j,rows), [=] (const int k) {
		Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,j,rows), [=] (const int k) {
			//Kokkos::single(Kokkos::PerThread(teamMember), [=] () {
			//Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,j+1,columns), [=] (const int l) {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j+1,columns), [=] (const int l) {
			//for (int l=j+1; l<columns; l++) {
				R(k,l) -= tau*t1(k)*t2(l);
			//}
			});
		});
		//teamMember.team_barrier();
		//Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
		//	Kokkos::single(Kokkos::PerThread(teamMember), [=] () {
		//		printf("tau %d: %f\n", j, t2(j));
		//	});
		//});
	}
	// copy to tau from t2 which was storing it in shared memory
	Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,0,columns), [=] (const int j) {
		tau(j) = t2(j);
	});

//	// demonstrates that R below diagonal, t1, and t2 contain all projector data along with taus
//	// and it works
//	for (int j = 0; j<columns; j++) {
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int k) {
////	Kokkos::single(Kokkos::PerThread(teamMember), [&] () {
//		// reuse t2 to be n by 1
//		// outer product of Q*w and tau*w
//
//			printf("%d %f\n",j, t1(j));
//			double t2k = Q(k,j)*t1(j);
//			for (int l=j+1; l<rows; l++) {
//				t2k += Q(k,l)*R(l,j);
//			}
//
//			Q(k,j) -= t2k*t2(j)*t1(j);
//			for (int l=j+1; l<rows; l++) {
//				Q(k,l) -= t2k*t2(j)*R(l,j);
//			}
//		});
////	});
//	}

	//if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
	//	// transpose R
	//	teamMember.team_barrier();
	//	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
	//		Kokkos::parallel_for(Kokkos::ThreadVectorRange(teamMember,0,i), [&] (const int j) {
	//			double tmp = R(i,j);
	//			R(i,j) = R(j,i);
	//			R(j,i) = tmp;
	//		});
	//	});
	//}
}

KOKKOS_INLINE_FUNCTION
double closestEigenvalueTwoByTwoEigSolver(const member_type& teamMember, double comparison_value, double a11, double a12, double a21, double a22) {
	/*
	 * Solves for eigenvalues of a 2x2 matrix and returns the one closest to a specified value
	 *
	 */

	double eigenvalue_relative_tolerance = 1e-15;
	double first_eig, second_eig;

	double b = -a11-a22;
	double c = a11*a22 - a21*a12;

	double sqrt_4ac = std::sqrt(4*c);
	if (4*c < 0) sqrt_4ac = 0;

	double bplus = b + sqrt_4ac;
	double bminus = b - sqrt_4ac;

	double discriminant = bplus*bminus;

	if (std::abs(a12) + std::abs(a21) < eigenvalue_relative_tolerance) { // check for a diagonal matrix

		first_eig = a11;
		second_eig = a22;

	} else if (discriminant < eigenvalue_relative_tolerance) { // check for a matrix where discriminant in quadratic formula introduces numerical error

		// 1. Perform inverse shifted power method with Rayleigh coefficient for cubic convergence
		// 2. Deflate by first eigenvalue
		// 2. Perform another inverse shifted power method for second eigenvalue

		double v[2] = {1, 1};
		double z[2] = {1, 1};
		double error = 1;
		double mu = 10;
		double norm_z;

		// Inverse shifted power method with Rayleigh coefficient
		while (error > eigenvalue_relative_tolerance) {

			norm_z = std::sqrt(z[0]*z[0] + z[1]*z[1]);
			v[0] = z[0] / norm_z;
			v[1] = z[1] / norm_z;

			double b11 = a11-mu, b12 = a12, b21 = a21, b22 = a22-mu;
			double c1 = (v[0] - b11*v[1]/b21)/(b12 - b11*b22/b21);
			double c2 = (v[1] - b22*c1) / b21;
			z[0] = c2;
			z[1] = c1;

			double lambda = z[0]*v[0] + z[1]*v[1];

			if (lambda==lambda)
				mu = mu + 1./lambda;
			else break;

			norm_z = std::sqrt(z[0]*z[0] + z[1]*z[1]);
			error = std::sqrt((z[0]-lambda*v[0])*(z[0]-lambda*v[0]) + (z[1]-lambda*v[1])*(z[1]-lambda*v[1])) / norm_z;

		}

		first_eig = mu;

		// Deflation by first eigenvalue found
		a11 -= first_eig*v[0]*v[0];
		a12 -= first_eig*v[0]*v[1];
		a21 -= first_eig*v[1]*v[0];
		a22 -= first_eig*v[1]*v[1];

		mu = a11 + a22;
		v[0] = 1; v[1] = 1;
		z[0] = 1; z[1] = 1;
		error = 1;

		// Inverse shifted power method with Rayleigh coefficient on the deflated matrix
		while (error > eigenvalue_relative_tolerance) {

			norm_z = std::sqrt(z[0]*z[0] + z[1]*z[1]);
			v[0] = z[0] / norm_z;
			v[1] = z[1] / norm_z;

			double b11 = a11-mu, b12 = a12, b21 = a21, b22 = a22-mu;
			double c1 = (v[0] - b11*v[1]/b21)/(b12 - b11*b22/b21);
			double c2 = (v[1] - b22*c1) / b21;
			z[0] = c2;
			z[1] = c1;

			double lambda = z[0]*v[0] + z[1]*v[1];

			if (lambda==lambda)
				mu = mu + 1./lambda;
			else break;

			norm_z = std::sqrt(z[0]*z[0] + z[1]*z[1]);
			error = std::sqrt((z[0]-lambda*v[0])*(z[0]-lambda*v[0]) + (z[1]-lambda*v[1])*(z[1]-lambda*v[1])) / norm_z;

		}

		second_eig = mu;

	} else {

		double sqrt_discriminant = std::sqrt(discriminant);
		first_eig = 2*c / (-b + sqrt_discriminant);
		second_eig = (-b + sqrt_discriminant) * 0.5;

	}

	double closest_eig = (std::abs(first_eig-comparison_value) < std::abs(second_eig-comparison_value)) ? first_eig : second_eig;
	return closest_eig;

}

KOKKOS_INLINE_FUNCTION
void orthogonalizeVectorBasis(const member_type& teamMember, scratch_matrix_type V) {

	// orthogonalize second vector against first
	double dot_product = V(0,0)*V(0,1) + V(1,0)*V(1,1) + V(2,0)*V(2,1);
	V(0,1) -= dot_product*V(0,0);
	V(1,1) -= dot_product*V(1,0);
	V(2,1) -= dot_product*V(2,0);

	double norm = std::sqrt(V(0,1)*V(0,1) + V(1,1)*V(1,1) + V(2,1)*V(2,1));
	V(0,1) /= norm;
	V(1,1) /= norm;
	V(2,1) /= norm;

	// orthogonalize third vector against second and first
	dot_product = V(0,0)*V(0,2) + V(1,0)*V(1,2) + V(2,0)*V(2,2);
	V(0,2) -= dot_product*V(0,0);
	V(1,2) -= dot_product*V(1,0);
	V(2,2) -= dot_product*V(2,0);

	norm = std::sqrt(V(0,2)*V(0,2) + V(1,2)*V(1,2) + V(2,2)*V(2,2));
	V(0,2) /= norm;
	V(1,2) /= norm;
	V(2,2) /= norm;

	dot_product = V(0,1)*V(0,2) + V(1,1)*V(1,2) + V(2,1)*V(2,2);
	V(0,2) -= dot_product*V(0,1);
	V(1,2) -= dot_product*V(1,1);
	V(2,2) -= dot_product*V(2,1);

	norm = std::sqrt(V(0,2)*V(0,2) + V(1,2)*V(1,2) + V(2,2)*V(2,2));
	V(0,2) /= norm;
	V(1,2) /= norm;
	V(2,2) /= norm;

}
KOKKOS_INLINE_FUNCTION
void largestTwoEigenvectorsThreeByThreeSymmetric(const member_type& teamMember, scratch_matrix_type V, scratch_matrix_type PtP, const int dimensions) {

	Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
		// put in a power method here and a deflation by first found eigenvalue
		double eigenvalue_relative_tolerance = 1e-6; // TODO: use something smaller, but really anything close is acceptable for this manifold
		double v[3] = {1, 1, 1};
		double v_old[3] = {1, 1, 1};

		double error = 1;
		double norm_v;

		while (error > eigenvalue_relative_tolerance) {

			double tmp1 = v[0];
			v[0] = PtP(0,0)*tmp1 + PtP(0,1)*v[1];
			if (dimensions>2) v[0] += PtP(0,2)*v[2];

			double tmp2 = v[1];
			v[1] = PtP(1,0)*tmp1 + PtP(1,1)*tmp2;
			if (dimensions>2) v[1] += PtP(1,2)*v[2];

			if (dimensions>2)
				v[2] = PtP(2,0)*tmp1 + PtP(2,1)*tmp2 + PtP(2,2)*v[2];

			norm_v = v[0]*v[0] + v[1]*v[1];
			if (dimensions>2) norm_v += v[2]*v[2];
			norm_v = std::sqrt(norm_v);

			v[0] = v[0] / norm_v;
			v[1] = v[1] / norm_v;
			if (dimensions>2) v[2] = v[2] / norm_v;

			error = (v[0]-v_old[0])*(v[0]-v_old[0]) + (v[1]-v_old[1])*(v[1]-v_old[1]);
			if (dimensions>2) error += (v[2]-v_old[2])*(v[2]-v_old[2]);
			error = std::sqrt(error);
			error /= norm_v;


			v_old[0] = v[0];
			v_old[1] = v[1];
			if (dimensions>2) v_old[2] = v[2];
		}

		double dot_product;
		double norm;

		// if 2D, orthogonalize second vector
		if (dimensions==2) {

			for (int i=0; i<2; ++i) {
				V(i,0) = v[i];
			}

			// orthogonalize second eigenvector against first
			V(0,1) = 1.0; V(1,1) = 1.0;
			dot_product = V(0,0)*V(0,1) + V(1,0)*V(1,1);
			V(0,1) -= dot_product*V(0,0);
			V(1,1) -= dot_product*V(1,0);

			norm = std::sqrt(V(0,1)*V(0,1) + V(1,1)*V(1,1));
			V(0,1) /= norm;
			V(1,1) /= norm;

		} else { // otherwise, work on second eigenvalue

			for (int i=0; i<3; ++i) {
				V(i,0) = v[i];
				for (int j=0; j<3; ++j) {
					PtP(i,j) -= norm_v*v[i]*v[j];
				}
			}

			error = 1;
			v[0] = 1; v[1] = 1; v[2] = 1;
			v_old[0] = 1; v_old[1] = 1; v_old[2] = 1;
			while (error > eigenvalue_relative_tolerance) {

				double tmp1 = v[0];
				v[0] = PtP(0,0)*tmp1 + PtP(0,1)*v[1] + PtP(0,2)*v[2];

				double tmp2 = v[1];
				v[1] = PtP(1,0)*tmp1 + PtP(1,1)*tmp2 + PtP(1,2)*v[2];

				v[2] = PtP(2,0)*tmp1 + PtP(2,1)*tmp2 + PtP(2,2)*v[2];

				norm_v = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

				v[0] = v[0] / norm_v;
				v[1] = v[1] / norm_v;
				v[2] = v[2] / norm_v;

				error = std::sqrt((v[0]-v_old[0])*(v[0]-v_old[0]) + (v[1]-v_old[1])*(v[1]-v_old[1]) + (v[2]-v_old[2])*(v[2]-v_old[2])) / norm_v;

				v_old[0] = v[0];
				v_old[1] = v[1];
				v_old[2] = v[2];
			}

			for (int i=0; i<3; ++i) {
				V(i,1) = v[i];
			}

			// orthogonalize second eigenvector against first
			dot_product = V(0,0)*V(0,1) + V(1,0)*V(1,1) + V(2,0)*V(2,1);

			V(0,1) -= dot_product*V(0,0);
			V(1,1) -= dot_product*V(1,0);
			V(2,1) -= dot_product*V(2,0);

			norm = std::sqrt(V(0,1)*V(0,1) + V(1,1)*V(1,1) + V(2,1)*V(2,1));
			V(0,1) /= norm;
			V(1,1) /= norm;
			V(2,1) /= norm;

			// orthogonalize third eigenvector against first and second
			V(0,2) = 1.0; V(1,2) = 1.0; V(2,2) = 1.0;
			dot_product = V(0,0)*V(0,2) + V(1,0)*V(1,2) + V(2,0)*V(2,2);
			V(0,2) -= dot_product*V(0,0);
			V(1,2) -= dot_product*V(1,0);
			V(2,2) -= dot_product*V(2,0);

			norm = std::sqrt(V(0,2)*V(0,2) + V(1,2)*V(1,2) + V(2,2)*V(2,2));
			V(0,2) /= norm;
			V(1,2) /= norm;
			V(2,2) /= norm;

			dot_product = V(0,1)*V(0,2) + V(1,1)*V(1,2) + V(2,1)*V(2,2);
			V(0,2) -= dot_product*V(0,1);
			V(1,2) -= dot_product*V(1,1);
			V(2,2) -= dot_product*V(2,1);

			norm = std::sqrt(V(0,2)*V(0,2) + V(1,2)*V(1,2) + V(2,2)*V(2,2));
			V(0,2) /= norm;
			V(1,2) /= norm;
			V(2,2) /= norm;

		}

	});

}

KOKKOS_INLINE_FUNCTION
void GivensBidiagonalReduction(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type U, scratch_matrix_type B, scratch_matrix_type V, const int columns, const int rows) {

	/*
	 * Performs a Bidiagonal reduction and overwrites the input matrix with the B matrix
	 * Stores and computes a full U and V matrix s.t. the original B = U*A*V' where A is the original B matrix passed in
	 * On GPUs, we store U and V transposed because data access is significantly faster on
	 * columns of a row rather than rows of a column.
	 *
	 * Regardless of device, the Layout of 2D data is not optimal for
	 * one of the two operations on B.
	 */
	const int target_index = teamMember.league_rank();

	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
		for(int j = 0; j < rows; ++j) {
			U(ORDER_INDICES(j,i)) = (i==j) ? 1 : 0;
		}
	});

	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int i) {
		for(int j = 0; j < columns; ++j) {
			V(ORDER_INDICES(j,i)) = (i==j) ? 1 : 0;
		}
	});

	teamMember.team_barrier();

	double c, s;
	for (int j=0; j<columns; ++j) {

		// for all of these (grabbing these two rows of be let's you calculate c, s
		for (int k=rows-2; k>=j; --k) {

			teamMember.team_barrier();
			GivensRotation(c, s, B(k,j), B(k+1,j));

			teamMember.team_barrier();

			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int l) {
				double tmp_val_1 = U(ORDER_INDICES(k,l));
				double tmp_val_2 = U(ORDER_INDICES(k+1,l));
				U(ORDER_INDICES(k,l)) = tmp_val_1*c + tmp_val_2*s;
				U(ORDER_INDICES(k+1,l)) = tmp_val_1*-s + tmp_val_2*c;
			});
			// not optimal for LayoutLeft
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, j, columns), [=] (const int l) {
				double tmp_val_1 = B(k,l);
				double tmp_val_2 = B(k+1,l);
				B(k,l) = c*tmp_val_1 + s*tmp_val_2;
				B(k+1,l) = -s*tmp_val_1 + c*tmp_val_2;
			});
		}

		for (int k=columns-2; k>j; --k) {

			teamMember.team_barrier();
			GivensRotation(c, s, B(j,k), B(j,k+1));

			teamMember.team_barrier();

			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int l) {
				double tmp_val_1 = V(ORDER_INDICES(k,l));
				double tmp_val_2 = V(ORDER_INDICES(k+1,l));
				V(ORDER_INDICES(k,l)) = c*tmp_val_1 + s*tmp_val_2;
				V(ORDER_INDICES(k+1,l)) = -s*tmp_val_1 + c*tmp_val_2;
			});
			// not optimal for LayoutRight
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j,rows), [=] (const int l) {
				double tmp_val_1 = B(l,k);
				double tmp_val_2 = B(l,k+1);
				B(l,k) = tmp_val_1*c + tmp_val_2*s;
				B(l,k+1) = tmp_val_1*-s + tmp_val_2*c;
			});
		}
	}
}

KOKKOS_INLINE_FUNCTION
void HouseholderBidiagonalReduction(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type U, scratch_matrix_type B, scratch_matrix_type V, const int columns, const int rows) {

	/*
	 * Performs a Bidiagonal reduction and overwrites the input matrix with the B matrix
	 * Stores and computes a full U and V matrix s.t. the original B = U*A*V' where A is the original B matrix passed in
	 */

	const int target_index = teamMember.league_rank();

	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
			for(int j = 0; j < rows; ++j) {
				U(i,j) = (i==j) ? 1 : 0;
			}
		});
	} else {
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
			for(int j = 0; j < rows; ++j) {
				U(j,i) = (i==j) ? 1 : 0;
			}
		});
	}

	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int i) {
			for(int j = 0; j < columns; ++j) {
				V(i,j) = (i==j) ? 1 : 0;
			}
		});
	} else {
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int i) {
			for(int j = 0; j < columns; ++j) {
				V(j,i) = (i==j) ? 1 : 0;
			}
		});
	}
	teamMember.team_barrier();

	for (int j = 0; j<columns; j++) {

		double normx = 0;
		// not optimized for LayoutRight
		Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,j,rows), [&] (const int i, double &temp_normx) {
			temp_normx += B(i,j)*B(i,j);
		}, normx);
		teamMember.team_barrier();
		normx = std::sqrt(normx);

		double s = (B(j,j) >= 0) ? -1 : 1; // flips sign R(j,j)
		double u1 = B(j,j) - s*normx;

		if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j+1,rows), [=] (const int i) {
				t1(i) = B(i,j) / u1;
			});
		} else {
			// not optimized for LayoutRight
			for (int i=j+1; i<rows; ++i) {
				t1(i) = B(i,j) / u1;
			}
		}
		t1(j) = 1;

		double tau = -s*u1/normx;

		teamMember.team_barrier();

		// outer product of tau*w and w'*R(j:end,:)
		// t1 is w, so use it to get w'*R(j:end,:)
		// t2 is 1 by n

		// potential candidate for parallel_scan
		if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
			// Not optimal for LayoutLeft
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int k) {
				double t2k = 0;
				for (int l=j; l<rows; l++) {
					t2k += t1(l)*B(l,k);
				}
				t2(k) = t2k;
			});
		} else {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int k) {
				double t2k = 0;
				for (int l=j; l<rows; l++) {
					t2k += t1(l)*B(l,k);
				}
				t2(k) = t2k;
			});
		}

		teamMember.team_barrier();
		if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j,rows), [=] (const int k) {
				for (int l=j; l<columns; l++) {
					// this ruins B being zero off the upper bidiagonal, but there is savings in not performing this elimination since we already knows what it does
					B(k,l) -= tau*t1(k)*t2(l);
				}
			});
		} else {
			// touches many columns of B
			for (int k=j; k<rows; k++) {
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int l) {
					// this ruins B being zero off the upper bidiagonal, but there is savings in not performing this elimination since we already knows what it does
					B(k,l) -= tau*t1(k)*t2(l);
				});
			}
		}

		if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int k) {
				double t2k = 0;
				for (int l=j; l<rows; l++) {
					t2k += U(k,l)*t1(l);
				}
				for (int l=j; l<rows; l++) {
					U(k,l) -= t2k*tau*t1(l);
				}
			});
		} else {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int k) {
				double t2k = 0;
				for (int l=j; l<rows; l++) {
					t2k += U(k,l)*t1(l);
				}
				t2(k) = t2k;
			});
			teamMember.team_barrier();
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int k) {
				for (int l=j; l<rows; l++) {
					U(k,l) -= t2(k)*tau*t1(l);
				}
			});
		}

		if (j <= columns-2) {

			normx = 0;
			Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,j+1,columns), [&] (const int i, double &temp_normx) {
				temp_normx += B(j,i)*B(j,i);
			}, normx);
			teamMember.team_barrier();
			normx = std::sqrt(normx);

			s = (B(j,j+1) >= 0) ? -1 : 1; // flips sign R(j,j)
			u1 = B(j,j+1) - s*normx;

			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,j+1,columns), [=] (const int i) {
				t1(i) = B(j,i) / u1;
			});
			t1(j+1) = 1;

			tau = -s*u1/normx;

			//// reuse t2 to be n by 1
			// outer product of Q*w and tau*w
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int k) {
				double t2k = 0;
				for (int l=j+1; l<columns; l++) {
					t2k += B(k,l)*t1(l);
				}
				t2(k) = t2k;
			});

			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int k) {
				for (int l=j+1; l<columns; l++) {
					B(k,l) -= t2(k)*tau*t1(l);
				}
			});

			if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
				//// reuse t2 to be n by 1
				// outer product of Q*w and tau*w
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int k) {
					double t2k = 0;
					for (int l=j+1; l<columns; l++) {
						t2k += V(k,l)*t1(l);
					}
					for (int l=j+1; l<columns; l++) {
						V(k,l) -= t2k*tau*t1(l);
					}
				});
			} else {
				//// reuse t2 to be n by 1
				// outer product of Q*w and tau*w
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int k) {
					double t2k = 0;
					for (int l=j+1; l<columns; l++) {
						t2k += V(k,l)*t1(l);
					}
					for (int l=j+1; l<columns; l++) {
						V(k,l) -= t2k*tau*t1(l);
					}
				});
			}
		}
	}

//	Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//		// checks for any nonzeros off of diagonal and superdiaganal entries
//		for (int k=0; k<m; k++) {
//			int min_k_n = (k<n) ? k : n;
//			for (int l=0; l<min_k_n; l++) {
//                                if (std::abs(B(k,l))>1e-13)
//				printf("%i %i %0.16f\n", k, l, B(k,l) );// << " " << l << " " << R(k,l) << std::endl;
//			}
//			for (int l=n-1; l>k+1; l--) {
//                                if (std::abs(B(k,l))>1e-13)
//				printf("%i %i %0.16f\n", k, l, B(k,l) );// << " " << l << " " << R(k,l) << std::endl;
//			}
//		}
//	});
}

KOKKOS_INLINE_FUNCTION
void GolubKahanSVD(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type U, scratch_matrix_type B, scratch_matrix_type V, const int B2b, const int B2e, const int columns, const int rows) {
	const int target_index = teamMember.league_rank();

	double c11=0, c12=0, c21=0, c22=0;

	int B2s = B2e - B2b;
	teamMember.team_barrier();

	Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,B2s), [&] (const int i, double &temp_val) {
		temp_val += B(i+B2b,B2e-2)*B(i+B2b,B2e-2);
	}, c11);
	Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,B2s), [&] (const int i, double &temp_val) {
		temp_val += B(i+B2b,B2e-2)*B(i+B2b,B2e-1);
	}, c12);
	Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,B2s), [&] (const int i, double &temp_val) {
		temp_val += B(i+B2b,B2e-1)*B(i+B2b,B2e-1);
	}, c22);

	teamMember.team_barrier();
	c21 = c12;

	double c22orc11 = (c22 != 0) ? c22 : c11;
	double closest_eig = closestEigenvalueTwoByTwoEigSolver(teamMember, c22orc11, c11, c12, c21, c22);

	if (std::abs(c11) + std::abs(c12) + std::abs(c22) < 1e-16) {
		closest_eig = 0;
	}

	int k=B2b;
	double alpha = B(k,k)*B(k,k) - closest_eig;
	double beta = B(k,k)*B(k,k+1);
	double c, s;

	for (k=B2b; k<B2e-1; ++k) {

		teamMember.team_barrier();

		GivensRotation(c, s, alpha, beta);

		// for each row of B perform get to column entry updates

		int minInd = (k-1 > -1) ? k-1 : 0;
		int maxInd = (k+2 < rows) ? k+2 : rows;

		teamMember.team_barrier();
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,maxInd-minInd), [=] (const int j) {
		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
			for (int j=0; j<maxInd-minInd; ++j) {
			const int j_ind = j + minInd;
			double tmp_val = B(j_ind,k);
			double tmp_val2 = B(j_ind,k+1);
			B(j_ind,k) = tmp_val*c + tmp_val2*s;
			B(j_ind,k+1) = tmp_val*-s + tmp_val2*c;
			}
		});

		if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int j) {
				double tmp_val = V(j,k);
				V(j,k) = c*tmp_val + s*V(j,k+1);
				V(j,k+1) = -s*tmp_val + c*V(j,k+1);
			});
		} else {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int j) {
				double tmp_val = V(k,j);
				V(k,j) = c*tmp_val + s*V(k+1,j);
				V(k+1,j) = -s*tmp_val + c*V(k+1,j);
			});
		}

		teamMember.team_barrier();

		alpha = B(k,k);
		beta = B(k+1,k);

		GivensRotation(c, s, alpha, beta);
		teamMember.team_barrier();

		maxInd = (k+3 < columns) ? k+3 : columns;
		teamMember.team_barrier();
//		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,maxInd-minInd), [=] (const int j) {
		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
			for (int j=0; j<maxInd-minInd; ++j) {
				const int j_ind = j + minInd;
				double tmp_val = B(k,j_ind);
				double tmp_val2 = B(k+1,j_ind);
				B(k,j_ind) = c*tmp_val + s*tmp_val2;
				B(k+1,j_ind) = -s*tmp_val + c*tmp_val2;
			}
		});

		if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int j) {
				double tmp_val = U(j,k);
				U(j,k) = tmp_val*c + U(j,k+1)*s;
				U(j,k+1) = tmp_val*-s + U(j,k+1)*c;
			});
		} else {
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int j) {
				double tmp_val = U(k,j);
				U(k,j) = tmp_val*c + U(k+1,j)*s;
				U(k+1,j) = tmp_val*-s + U(k+1,j)*c;
			});
		}

		teamMember.team_barrier();
		if (k < B2e-2) {
			alpha = B(k,k+1);
			beta = B(k,k+2);
		}

		teamMember.team_barrier();
	}

	teamMember.team_barrier();

}

KOKKOS_INLINE_FUNCTION
void GolubReinschSVD(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type B, scratch_matrix_type U,  scratch_vector_type S, scratch_matrix_type V, const int columns, const int rows) {
	const int target_index = teamMember.league_rank();

//	Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//		// checks for any nonzeros off of diagonal and superdiaganal entries
//		for (int k=0; k<rows; k++) {
//			for (int l=0; l<columns; l++) {
//				if (B(k,l)!=0 && target_index==0)
//				printf("%i %i %0.16f\n", k, l, B(k,l) );// << " " << l << " " << R(k,l) << std::endl;
//			}
//		}
//	});
//
//	printf("after:\n");

//      // Without transposing U and V, Householder is faster on GPUs.
//      // With transposing U and V, Givens is faster on GPUs.
//	BidiagonalReduction requires eliminating rows and columns which is not optimal
//      for either matrix layout (Left or Right) approximately half of the time

//	The matrix B was filled in transposed because this is more efficient for QR and other algorithms
//	but must be retransposed for this algorithm
	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
		// transpose R because layout not optimal as LayoutLeft for row operations
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
			for(int j = 0; j < i; ++j) {
				double tmp = B(i,j);
				B(i,j) = B(j,i);
				B(j,i) = tmp;
			}
		});
		teamMember.team_barrier();
	}
#ifdef COMPADRE_USE_CUDA
	GMLS_LinearAlgebra::HouseholderBidiagonalReduction(teamMember, t1, t2, U, B, V, columns, rows); // perform bidiagonal reduction
#else
	GMLS_LinearAlgebra::GivensBidiagonalReduction(teamMember, t1, t2, U, B, V, columns, rows); // perform bidiagonal reduction
#endif

//	 // diagnostic for U*B*V' to check against A
//	Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//		// checks for any nonzeros off of diagonal and superdiagonal entries
//		for (int i=0; i<m; i++) {
//			for (int j=0; j<n; j++) {
//				double val = 0;
//				for (int k=0; k<m; k++) {
//					for (int l=0; l<n; l++) {
//						val += U(i,k)*B(k,l)*V(j,l);
//					}
//				}
//				printf("%i %i %0.16f\n", i, j, val );// << " " << l << " " << R(k,l) << std::endl;
//			}
//		}
//	});

	int B3s = 0;
	int count = 0;

	double offdiagonal_tolerance = 1e-14;
	double absolute_offdiagonal_tolerance = 1e-16;
	double diagonal_tolerance = 1e-18;

	while (B3s != columns) {

		count++;
		teamMember.team_barrier();

		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns-1), [=] (const int i) {
			if (std::abs(B(i,i+1)) < absolute_offdiagonal_tolerance || std::abs(B(i,i+1)) < offdiagonal_tolerance * (std::abs(B(i,i)) + std::abs(B(i+1,i+1)))) B(i,i+1) = 0;
		});
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns-1), [=] (const int i) {
			if (std::abs(B(i+1,i)) < absolute_offdiagonal_tolerance || std::abs(B(i+1,i)) < offdiagonal_tolerance * (std::abs(B(i,i)) + std::abs(B(i+1,i+1)))) B(i+1,i) = 0;
		});
//		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//			for (int i=0; i<n-1; ++i) {
//				if (std::abs(B(i,i+1)) < offdiagonal_tolerance2 * (std::abs(B(i,i)) + std::abs(B(i+1,i+1)))) B(i,i+1) = 0;
//				if (std::abs(B(i+1,i)) < offdiagonal_tolerance2 * (std::abs(B(i,i)) + std::abs(B(i+1,i+1)))) B(i+1,i) = 0;
//			}
//		});

		teamMember.team_barrier();

		// we break the problem up into matrices B1, B2, and B3 with beginning indices B1b, B2b, B3b and ending indices (exclusive)
		// of B1e, B2e, B3e
		// we do not actually do any work on B1 so there is no point in calculating its size, beginning, or end
		int B2b, B3b, B2e, B3e=columns;

		// first we find B3 by which will end at the bottom right and is as large as possible so that we end up with B3 that is diagonal
		// i.e. B=[B1 0 0; 0 B2 0; 0 0 B3] where B3 is the largest diagonal matrix possible, and B2 is the largest matrix possible
		// following the determination of B3 that will allow B2 to have nonzero superdiagonal.

		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
			// serial work so we use team visible vector to write to
			// with the following encoding:
			t1(0) = 0; // B2b
			t1(1) = 0; // B3b

			for (int i=columns-1; i>0; --i) {
				if (std::abs(B(i-1,i)) != 0) {
					// reached a nonzero superdiagonal entry
					t1(1) = i+1; // B3b
					break;
				}
			}

			// B2 ends where B3 begins
			for (int i=t1(1)-1; i>0; --i) {
				if (std::abs(B(i-1,i)) == 0) {
					t1(0) = i; // B2b
					break;
				}
			}
		});
		teamMember.team_barrier();

		// make sure every thread now has access to these values
		B2b = t1(0);
		B3b = t1(1);
		B2e = B3b;

		// size of Block 3
		B3s = B3e-B3b;

		if (B3s == columns) {
			break;
		}
		// all work to be done is on B2. If we made it here, then there is some work to be done.
		// start by checking for diagonal zero entries


		bool work_on_zero_on_diagonal = false;
		for (int i=B2b; i<B2e; ++i) {
			if (std::abs(B(i,i)) < diagonal_tolerance) {
				double c, s;
				if (i+1 < columns) { // can't perform if out of bounds

					int minInd = (i-1 > -1) ? i-1 : 0;
					int maxInd = (i+2 < rows) ? i+2 : rows;
					if (std::abs(B(i,i+1)) > absolute_offdiagonal_tolerance) {
						GivensRotation(c, s, B(i,i), B(i,i+1));

						// for each row of B perform get to column entry updates
						Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
							double tmp_val = 0;
							for (int j=minInd; j<maxInd; ++j) {
								tmp_val = B(j,i);
								B(j,i) = tmp_val*c + B(j,i+1)*s;
								B(j,i+1) = tmp_val*-s + B(j,i+1)*c;
							}

							if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
								for (int j=0; j<columns; ++j) {
									tmp_val = V(j,i);
									V(j,i) = c*tmp_val + s*V(j,i+1);
									V(j,i+1) = -s*tmp_val + c*V(j,i+1);
								}
							} else {
								// Working on transposed V
								for (int j=0; j<columns; ++j) {
									tmp_val = V(i,j);
									V(i,j) = c*tmp_val + s*V(i+1,j);
									V(i+1,j) = -s*tmp_val + c*V(i+1,j);
								}
							}
						});
						work_on_zero_on_diagonal = true;
					}
					maxInd = (i+3 < columns) ? i+3 : columns;
					teamMember.team_barrier();
					if (std::abs(B(i+1,i)) > absolute_offdiagonal_tolerance) {
						GivensRotation(c, s, B(i,i), B(i+1,i));

						Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
							double tmp_val = 0;
							// for each col of B perform get to column entry updates
							for (int j=minInd; j<maxInd; ++j) {
								tmp_val = B(i,j);
								B(i,j) = c*tmp_val + s*B(i+1,j);
								B(i+1,j) = -s*tmp_val + c*B(i+1,j);
							}
							if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutLeft>::value) {
								for (int j=0; j<rows; ++j) {
									tmp_val = U(j,i);
									U(j,i) = tmp_val*c + U(j,i+1)*s;
									U(j,i+1) = tmp_val*-s + U(j,i+1)*c;
								}
							} else {
								// working on transposed U
								for (int j=0; j<rows; ++j) {
									tmp_val = U(i,j);
									U(i,j) = tmp_val*c + U(i+1,j)*s;
									U(i+1,j) = tmp_val*-s + U(i+1,j)*c;
								}
							}
						});
						work_on_zero_on_diagonal = true;
					}
				}
			}
		}

		if (!work_on_zero_on_diagonal) {

			GolubKahanSVD(teamMember, t1, t2, U, B, V, B2b, B2e, columns, rows);
		}
//		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//			printf("count: %d, B3s%d B3e%d B3b%d\n", count, B3s, B3e, B3b);
////				if (target_index==8) {
//				for (int k=0; k<m; k++) {
//					for (int l=0; l<n; l++) {
//						if (std::abs(B(k,l)) > diagonal_tolerance)
//							printf("%i %i %0.16f\n", k, l, B(k,l) );
//					}
//				}
//		if (count > 1000)
//			printf("Zero on diagonal, %d, %d\n", target_index, B3s);
////				}
//		});
//		teamMember.team_barrier();
//		printf("%d\n", count);
	}

//		 // diagnosticG for U*B*V' to check against A
//		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//			// checks for any nonzeros off of diagonal and superdiagonal entries
//			for (int i=0; i<m; i++) {
//				for (int j=0; j<n; j++) {
//					double val = 0;
//					for (int k=0; k<m; k++) {
//						for (int l=0; l<n; l++) {
//							val += U(i,k)*B(k,l)*V(j,l);
//						}
//					}
//					printf("%i %i %0.16f\n", i, j, val );// << " " << l << " " << R(k,l) << std::endl;
//				}
//			}
//		});

//		 // diagnosticG for U'*U  and V*V' to check for orthogonality and unitary
//		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//			// checks for any nonzeros off of diagonal and superdiagonal entries
//			for (int i=0; i<n; i++) {
//				for (int j=0; j<n; j++) {
//					double val = 0;
////					for (int k=0; k<m; k++) {
////						val += U(k,i)*U(k,j);
////					}
//					for (int k=0; k<n; k++) {
//						val += V(k,i)*V(k,j);
//					}
////					if (i!=j && std::abs(val)>1e-11 )
//					if (i==j)
//						printf("%i %i %0.16f\n", i, j, val );// << " " << l << " " << R(k,l) << std::endl;
//				}
//			}
//		});

//	}
//		if (target_index == 0) {
//		Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//			for (int k=0; k<m; k++) {
//				for (int l=0; l<n; l++) {
////					if (std::abs(B(k,l)) > diagonal_tolerance)
//						printf("%i %i %0.16f\n", k, l, B(k,l) );
//				}
//			}
//		}); }
		// V contains eigenvectors of P'*P

//		Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
//			for (int i=0; i<S.size(); i++) {
//				S(i) = PsqrtW(i,i);
//			}
//		});
//		teamMember.team_barrier();
//	}
//	Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//		printf("converged in %d iterations. \n", count);
//	});
	teamMember.team_barrier();

	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) {
		// retranspose V and U
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,rows), [=] (const int i) {
			for(int j = 0; j < i; ++j) {
				double tmp = U(i,j);
				U(i,j) = U(j,i);
				U(j,i) = tmp;
			}
		});
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int i) {
			for(int j = 0; j < i; ++j) {
				double tmp = V(i,j);
				V(i,j) = V(j,i);
				V(j,i) = tmp;
			}
		});
		teamMember.team_barrier();
	}

	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,columns), [=] (const int i) {
		S(i) = B(i,i);
	});
	teamMember.team_barrier();



//	Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//		// checks for any nonzeros off of diagonal and superdiaganal entries
//		for (int k=0; k<m; k++) {
//			for (int l=0; l<n; l++) {
//				if (B(k,l)!=0 && target_index==0)
//				printf("%i %i %0.16f\n", k, l, B(k,l) );// << " " << l << " " << R(k,l) << std::endl;
//			}
//		}
//	});
}

KOKKOS_INLINE_FUNCTION
void matrixToLayoutLeft(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type A, const int columns, const int rows) {
	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) {

		int dim_0 = A.dimension_0();
		int dim_1 = A.dimension_1();
		int block_size = std::floor((double)(std::sqrt(t1.dimension_0())));
		int col_blocks = std::ceil((double)(columns) / block_size);
		int row_blocks = std::ceil((double)(rows) / block_size);
		int total_blocks = col_blocks * row_blocks;

		// blocks go left to right, top to bottom
		
		// read in by blocks
		for (int block_num = 0; block_num < total_blocks; ++block_num) {

			int row_block = block_num / col_blocks;
			int col_block = block_num % col_blocks;

			if (row_block >= col_block) {

				int i_offset  = row_block * block_size;
				int j_offset  = col_block * block_size;

				int local_block_width  = ((columns - j_offset) < block_size) ? columns - j_offset : block_size;
				int local_block_height = ((rows - i_offset) < block_size) ? rows - i_offset : block_size;

				// read in block from A
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,local_block_width), [=] (const int j) {
					for(int i = 0; i < local_block_height; ++i) {
						// read a contiguous column of A into a contiguous section of t1
						t1(i*block_size + j) = *(A.data() +  (i_offset+i)*dim_1 + (j_offset+j)); // fast read into t1
					}
				});
				teamMember.team_barrier();

				// read in block to be written to, now stored in t1
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,local_block_height), [=] (const int i) {
					for(int j = 0; j < local_block_width; ++j) {
						// read a contiguous column of A into a contiguous section of t1
						t2(i*block_size + j) = *(A.data() +  (j_offset+j)*dim_0 + (i_offset+i)); // fast read into t1
					}
				});
				teamMember.team_barrier();

				// write into A from t2
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,local_block_width), [=] (const int j) {
					for(int i = 0; i < local_block_height; ++i) {
						// read a contiguous column of A into a contiguous section of t1
						*(A.data() +  (i_offset+i)*dim_1 + (j_offset+j)) = t2(i*block_size + j); // fast read from t2
					}
				});
				teamMember.team_barrier();

				// write into A from t1
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,local_block_height), [=] (const int i) {
					for(int j = 0; j < local_block_width; ++j) {
						// read a contiguous column of A into a contiguous section of t1
						*(A.data() +  (j_offset+j)*dim_0 + (i_offset+i)) = t1(i*block_size + j);
					}
				});
				teamMember.team_barrier();
			}
			teamMember.team_barrier();
		}
		teamMember.team_barrier();
	}
}

//void batchQRFactorize(double *P, double *RHS, const size_t dim_0, const size_t dim_1, const int num_matrices) {
//#ifdef COMPADRE_USE_CUDA
//
//    Kokkos::Profiling::pushRegion("QR::Setup(Pointers)");
//    Kokkos::View<size_t*> array_P_RHS("P and RHS matrix pointers on device", 2*num_matrices);
//    // get pointers to device data
//    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA(const int i) {
//        array_P_RHS(i               ) = reinterpret_cast<size_t>(P   + i*dim_0*dim_0);
//        array_P_RHS(i + num_matrices) = reinterpret_cast<size_t>(RHS + i*dim_0*dim_0);
//    });
//    int *devInfo; cudaMalloc(&devInfo, num_matrices*sizeof(int));
//    cudaDeviceSynchronize();
//    Kokkos::Profiling::popRegion();
//
//    Kokkos::Profiling::pushRegion("QR::Setup(Handle)");
//    // Create cublas instance
//    cublasHandle_t cublas_handle;
//    cublasStatus_t cublas_stat;
//    cudaDeviceSynchronize();
//    Kokkos::Profiling::popRegion();
//
//    Kokkos::Profiling::pushRegion("QR::Setup(Create)");
//    cublasCreate(&cublas_handle);
//    cudaDeviceSynchronize();
//    Kokkos::Profiling::popRegion();
//
//    Kokkos::Profiling::pushRegion("QR::Execution");
//    // call batched QR
//    int info;
//    //cublas_stat=cublasDgeqrfBatched(cublas_handle,
//    //                         static_cast<int>(dim_0), static_cast<int>(dim_1), reinterpret_cast<double**>(array_P_Q_tau.ptr_on_device()),
//    //                         static_cast<int>(dim_0),
//    //                         reinterpret_cast<double**>(array_P_Q_tau.ptr_on_device()+num_matrices),
//    //                         &info, num_matrices);
//    cublas_stat=cublasDgelsBatched(cublas_handle,
//                                   CUBLAS_OP_N, 
//                                   static_cast<int>(dim_0),  // m
//                                   static_cast<int>(dim_1),  // n
//                                   static_cast<int>(dim_0),  // nrhs
//                                   reinterpret_cast<double**>(array_P_RHS.ptr_on_device()),
//                                   static_cast<int>(dim_0), // lda
//                                   reinterpret_cast<double**>(array_P_RHS.ptr_on_device() + num_matrices),
//                                   static_cast<int>(dim_0), // ldc
//                                   &info, 
//                                   devInfo,
//                                   num_matrices );
//
//    if (cublas_stat != CUBLAS_STATUS_SUCCESS)
//      printf("\n cublasDgeqrfBatched failed");
//    cudaDeviceSynchronize();
//    Kokkos::Profiling::popRegion();
//
//
//#elif defined(COMPADRE_USE_LAPACK)
//
//    // requires column major input
//
//    Kokkos::Profiling::pushRegion("QR::Setup(Create)");
//
//    // find optimal blocksize when using LAPACK in order to allocate workspace needed
//    int ipsec = 1, unused = -1, bnp = static_cast<int>(dim_1), lwork = -1, info = 0; double wkopt = 0;
//    int mmd = static_cast<int>(dim_0);
//    dgels_( (char *)"N", &mmd, &bnp, &mmd, 
//            (double *)NULL, &mmd, 
//            (double *)NULL, &mmd, 
//            &wkopt, &lwork, &info );
//
//    // size needed to malloc for each problem
//    lwork = (int)wkopt;
//    printf("%d is opt: lapack_opt_blocksize\n", lwork);
//
//    double *work = (double *)malloc(num_matrices*lwork*sizeof(double));
//    Kokkos::View<int*> infos("info", num_matrices);
//    Kokkos::deep_copy(infos, 0);
//    Kokkos::Profiling::popRegion();
//
//    // get OMP_NUM_THREADS to store as a reference, then set to 1
////#if defined(_OPENMP)
////	unsigned int thread_qty;
////    try {
////        std::string omp_string = std::getenv("OMP_NUM_THREADS");
////    	thread_qty = std::max(atoi(omp_string.c_str()), 1);
////	} catch (...) {
////    	thread_qty = 1;
////	}
////    omp_set_num_threads(1);
////    printf("%d threads called.\n", thread_qty);
////#endif
////
////#ifdef Compadre_USE_OPENBLAS
////   int openblas_num_threads = 1;
////   set_openblas_num_threads(openblas_num_threads);
////   set_openmp_num_threads(openblas_num_threads);
////   printf("Reduce openblas threads to 1.\n");
////#endif
//
//	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA (const int i) {
//                int t_dim_0 = dim_0;
//                int t_dim_1 = dim_1;
//                int t_info = infos[i];
//                int t_lwork = lwork;
//                double * p_offset = P + i*dim_0*dim_0;
//                double * rhs_offset = RHS + i*dim_0*dim_0;
//                double * work_offset = work + i*lwork;
//                dgels_( (char *)"N", &t_dim_0, &t_dim_1, &t_dim_0, 
//                        p_offset, &t_dim_0, 
//                        rhs_offset, &t_dim_0, 
//                        work_offset, &t_lwork, &t_info);
//	});
//
//
//    Kokkos::Profiling::pushRegion("QR::Setup(Cleanup)");
//    free(work);
//    Kokkos::Profiling::popRegion();
//
//#endif
//
//}


//void batchLUFactorize(double *P, double *RHS, const size_t dim_0, const size_t dim_1, const int num_matrices) {
//}

};

#endif

