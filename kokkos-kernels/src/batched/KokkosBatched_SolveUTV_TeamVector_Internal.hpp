#ifndef __KOKKOSBATCHED_SOLVE_UTV_TEAMVECTOR_INTERNAL_HPP__
#define __KOKKOSBATCHED_SOLVE_UTV_TEAMVECTOR_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_Gemv_TeamVector_Internal.hpp"
#include "KokkosBatched_Trsv_TeamVector_Internal.hpp"

#include "KokkosBatched_Gemm_TeamVector_Internal.hpp"
#include "KokkosBatched_Trsm_TeamVector_Internal.hpp"

namespace KokkosBatched {

  ///
  /// TeamVector Internal
  /// =================== 
  struct TeamVectorSolveUTV_Internal {
    template<typename MemberType,
             typename ValueType,
	     typename IntType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
	   const int matrix_rank,
           const int m, const int n,
           const ValueType * U, const int us0, const int us1,
	   const ValueType * T, const int ts0, const int ts1,
	   const ValueType * V, const int vs0, const int vs1,
	   const IntType   * p, const int ps0,
	   /* */ ValueType * x, const int xs0,
	   /* */ ValueType * b, const int bs0,
           /* */ ValueType * w) {
      typedef ValueType value_type;
      //typedef IntType int_type;

      const value_type one(1), zero(0);
      const int ws0 = 1;

      if (matrix_rank < m) {
	/// w = U^T b
	TeamVectorGemvInternal<Algo::Gemv::Unblocked>
	  ::invoke(member,
		   matrix_rank, m,
		   one,
		   U, us1, us0,
		   b, bs0,
		   zero,
		   w, ws0);
	
	/// w = T^{-1} w
	TeamVectorTrsvInternalLower<Algo::Trsv::Unblocked>
	  ::invoke(member,
		   false,
		   matrix_rank,
		   one,
		   T, ts0, ts1,
		   w, ws0);
	
	/// x = V^T w
	TeamVectorGemvInternal<Algo::Gemv::Unblocked>
	  ::invoke(member,
		   m, matrix_rank, 
		   one,
		   V, vs1, vs0,
		   w, ws0,
		   zero,
		   x, xs0);
      } else {
	TeamVectorGemvInternal<Algo::Gemv::Unblocked>
	  ::invoke(member,
		   matrix_rank, m,
		   one,
		   U, us1, us0,
		   b, bs0,
		   zero,
		   x, xs0);

	TeamVectorTrsvInternalUpper<Algo::Trsv::Unblocked>
	  ::invoke(member,
		   false,
		   matrix_rank,
		   one,
		   T, ts0, ts1,
		   x, xs0);	
      }

      /// x = P^T x
      TeamVectorApplyPivotVectorBackwardInternal
	::invoke(member,
		 m,
		 p, ps0,
		 x, xs0);

      return 0;
    }

    template<typename MemberType,
             typename ValueType,
	     typename IntType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
	   const int matrix_rank,
           const int m, const int n, const int nrhs,
           const ValueType * U, const int us0, const int us1,
	   const ValueType * T, const int ts0, const int ts1,
	   const ValueType * V, const int vs0, const int vs1,
	   const IntType   * p, const int ps0,
	   /* */ ValueType * X, const int xs0, const int xs1,
	   /* */ ValueType * B, const int bs0, const int bs1,
           /* */ ValueType * w) {
      typedef ValueType value_type;
      //typedef IntType int_type;

      const value_type one(1), zero(0);

      value_type * W = w; /// m x nrhs
      const int ws0 = xs0 < xs1 ? 1 : nrhs, ws1 = xs0 < xs1 ? m : 1;

      bool do_print = false;
      if (do_print) {
printf("size is: %d %d %d %d\n", matrix_rank, m, n, nrhs); 
printf("U, us1, us0: %d %d\n", us1, us0);
printf("T, ts0, ts1: %d %d\n", ts0, ts1);
printf("B, bs0, bs1: %d %d\n", bs0, bs1);
printf("X, xs0, xs1: %d %d\n", xs0, xs1);
printf("W, ws0, ws1: %d %d\n", ws0, ws1);
      }

      if (matrix_rank < n) {
          if (do_print) printf("went here\n");
	// U is m x matrix_rank
	// T is matrix_rank x matrix_rank
	// V is matrix_rank x n
	// W = U^T B
    //for (int i=0; i<m; ++i) {
    //    for (int j=0; j<nrhs; ++j) {
    //        W[i*ws0+j]=0.0;
    //    }
    //}
	TeamVectorGemmInternal<Algo::Gemm::Unblocked>
	  ::invoke(member,
		   matrix_rank, nrhs, m,
		   one,
		   U, us1, us0,
		   B, bs0, bs1,
		   zero,
		   W, ws0, ws1);
    //member.team_barrier();
      if (do_print) {
      printf("W=zeros(%d,%d);\n", m, nrhs);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<nrhs; ++j) {
            printf("W(%d,%d)= %f;\n", i+1,j+1,W[i*ws0+j]);
        }
    }
      printf("B=zeros(%d,%d);\n", m, nrhs);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<nrhs; ++j) {
            printf("B(%d,%d)= %f;\n", i+1,j+1,B[i*bs0+j]);
        }
    }
      }

	/// W = T^{-1} W
	TeamVectorTrsmInternalLeftLower<Algo::Trsm::Unblocked>
	  ::invoke(member,
		   false,
		   matrix_rank, nrhs,
		   one,
		   T, ts0, ts1,
		   W, ws0, ws1);

      if (do_print) {
      printf("W=zeros(%d,%d);\n", m, nrhs);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<nrhs; ++j) {
            printf("W(%d,%d)= %f;\n", i+1,j+1,W[i*ws0+j]);
        }
    }
      }
	
	/// X = V^T W
	TeamVectorGemmInternal<Algo::Gemm::Unblocked>
	  ::invoke(member,
		   n, nrhs, matrix_rank, 
		   one,
		   V, vs1, vs0,
		   W, ws0, ws1,
		   zero,
		   X, xs0, xs1);
      if (do_print) {
      printf("X=zeros(%d,%d);\n", n, nrhs);
    for (int i=0; i<n; ++i) {
        for (int j=0; j<nrhs; ++j) {
            printf("X(%d,%d)= %f;\n", i+1,j+1,X[i*xs0+j]);
        }
    }
      }
      } else {
          if (do_print) printf("ELSE\n");
	/// W = U^T B
	TeamVectorGemmInternal<Algo::Gemm::Unblocked>
	  ::invoke(member,
		   matrix_rank, nrhs, m, 
		   one,
		   U, us1, us0,
		   B, bs0, bs1,
		   zero,
		   X, xs0, xs1);

      if (do_print) {
    printf("m=zeros(%d,%d);\n", matrix_rank, nrhs);
    for (int i=0; i<matrix_rank; ++i) {
        for (int j=0; j<nrhs; ++j) {
            printf("m(%d,%d)= %f;\n", i+1,j+1,X[i*ws0+j]);
        }
    }
      }

    //// copy W -> X
    //for (int i=0; i<matrix_rank; ++i) {
    //    for (int j=0; j<nrhs; ++j) {
    //        X[i*ws0+j] = W[i*ws0+j];
    //    }
    //}
    //for (int i=matrix_rank; i<n; ++i) {
    //    for (int j=0; j<nrhs; ++j) {
    //        X[i*ws0+j] = 0;
    //    }
    //}
      if (do_print) {
      printf("T=zeros(%d,%d);\n", m, matrix_rank);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<matrix_rank; ++j) {
            printf("T(%d,%d)= %f;\n", i+1,j+1,T[i*ts0+j]);
        }
    }
      }

	/// X = T^{-1} X
	TeamVectorTrsmInternalLeftUpper<Algo::Trsm::Unblocked>
	  ::invoke(member,
		   false,
		   matrix_rank, nrhs,
		   one,
		   T, ts0, ts1,
		   X, xs0, xs1);		
    //for (int i=matrix_rank; i<n; ++i) {
    //    for (int j=0; j<nrhs; ++j) {
    //        X[i*ws0+j] = 0;
    //    }
    //}
      if (do_print) {
    printf("x=zeros(%d,%d);\n", n, nrhs);
    for (int i=0; i<n; ++i) {
        for (int j=0; j<nrhs; ++j) {
            printf("x(%d,%d)= %f;\n", i+1,j+1,X[i*xs0+j]);
        }
    }
      }




    //TeamVectorCopy<Trans::NoTranspose>::invoke(member, W, X);
    //  TeamVectorCopy<MemberType,Trans::NoTranspose>
	//::invoke(member, W, matrix_rank, nrhs, ws0, ws1, X, xs0, xs1);
    //KOKKOSBATCHED_TEAM_COPY_MATRIX_NO_TRANSPOSE_INTERNAL_INVOKE(member, W, matrix_rank, nrhs, ws0, ws1, X, xs0, xs1);


	///// X = V^T W
	//TeamVectorGemmInternal<Algo::Gemm::Unblocked>
	//  ::invoke(member,
	//	   n, nrhs, matrix_rank, 
	//	   one,
	//	   V, vs1, vs0,
	//	   W, ws0, ws1,
	//	   zero,
	//	   X, xs0, xs1);
    }
      
      ///// X = P^T X
      TeamVectorApplyPivotMatrixBackwardInternal
      	::invoke(member,
      		 nrhs, matrix_rank,
      		 p, ps0,
      		 X, xs0, xs1);
      if (do_print) printf("done!\n");

      return 0;
    }

  };

} // end namespace KokkosBatched


#endif
