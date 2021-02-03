#ifndef __KOKKOSBATCHED_UTV_TEAMVECTOR_INTERNAL_HPP__
#define __KOKKOSBATCHED_UTV_TEAMVECTOR_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_SetTriangular_Internal.hpp"
#include "KokkosBatched_QR_TeamVector_Internal.hpp"
#include "KokkosBatched_QR_WithColumnPivoting_TeamVector_Internal.hpp"
#include "KokkosBatched_QR_FormQ_TeamVector_Internal.hpp"

namespace KokkosBatched {

  ///
  /// TeamVector Internal
  /// =================== 
  struct TeamVectorUTV_Internal {
    template<typename MemberType,
             typename ValueType,
	     typename IntType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const int m, const int n, // m = NumRows(A)
           /* */ ValueType * A, const int as0, const int as1,
	   /* */ IntType   * p, const int ps0,
	   /* */ ValueType * U, const int us0, const int us1,
	   /* */ ValueType * V, const int vs0, const int vs1,
           /* */ ValueType * w, // 3*m, tau, norm, householder workspace
	   /* */ int &matrix_rank) {
      typedef ValueType value_type;
      //typedef IntType int_type;

      value_type *t = w; w+= m;
      const int ts0(1);

      value_type *work = w;

      bool do_print=false;

      if (do_print) {
        Kokkos::single(Kokkos::PerTeam(member), [&] () {
      printf("A=zeros(%d,%d);\n", m, n);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            printf("A(%d,%d)= %f;\n", i+1,j+1,A[i*as0+j]);
        }
    }
        });
      }

      matrix_rank = -1;
      TeamVectorQR_WithColumnPivotingInternal
      	::invoke(member,
      		 m, n,
      		 A, as0, as1,
      		 t, ts0,
      		 p, ps0,
      		 work,
      		 matrix_rank);
      if (do_print) {
        Kokkos::single(Kokkos::PerTeam(member), [&] () {
      printf("R=zeros(%d,%d);\n", m, matrix_rank);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            printf("R(%d,%d)= %f;\n", i+1,j+1,A[i*as0+j]);
        }
    }
      printf("P=zeros(%d);\n", n);
    for (int i=0; i<n; ++i) {
        printf("P(%d)= %d;\n", i+1,p[i]);
    }
      printf("t=zeros(%d);\n", m);
    for (int i=0; i<m; ++i) {
        printf("t(%d)= %f;\n", i+1,t[i]);
    }
      printf("w=zeros(%d);\n", m);
    for (int i=0; i<m; ++i) {
        printf("w(%d)= %f;\n", i+1,w[i]);
    }
    printf("m(%d), n(%d), matrix_rank(%d)\n", m, n, matrix_rank);
        });
      }

      TeamVectorQR_FormQ_Internal
      	::invoke(member,
      		 m, matrix_rank, matrix_rank,
      		 A, as0, as1,
      		 t, ts0,
      		 U, us0, us1,
      		 work);
      member.team_barrier();

      if (do_print) {
        Kokkos::single(Kokkos::PerTeam(member), [&] () {
      printf("R=zeros(%d,%d);\n", m, matrix_rank);
    for (int i=0; i<m; ++i) {
        for (int j=i; j<n; ++j) {
            printf("R(%d,%d)= %f;\n", i+1,j+1,A[i*as0+j]);
        }
    }
      printf("U=zeros(%d,%d);\n", m, m);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<m; ++j) {
            printf("U(%d,%d)= %f;\n", i+1,j+1,U[i*us0+j]);
        }
    }
    printf("m(%d), n(%d), matrix_rank(%d)\n", m, n, matrix_rank);
        });
      }
      

      /// for rank deficient matrix
      if (matrix_rank < n) {

	const value_type zero(0);

    // triu
	TeamVectorSetLowerTriangularInternal
	  ::invoke(member,
		   matrix_rank, matrix_rank,
		   1, zero,
		   A, as0, as1);
      if (do_print) {
        Kokkos::single(Kokkos::PerTeam(member), [&] () {
    printf("t=zeros(%d,%d);\n", m, n);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            printf("T(%d,%d)= %f;\n", i+1,j+1,A[i*as0+j]);
        }
    }
        });
      }
	
    // A' now contains [q,r]=qr(triu(R'))
	TeamVectorQR_Internal
	  ::invoke(member,
		   n, matrix_rank,
		   A, as1, as0,
		   t, ts0,
		   work);
      if (do_print) {
          Kokkos::single(Kokkos::PerTeam(member), [&] () {
    printf("L=zeros(%d,%d);\n", m, n);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            printf("L(%d,%d)= %f;\n", i+1,j+1,A[i*as0+j]);
        }
    }
    for (int i=0; i<n; ++i) {
        printf("tau(%d): %f\n", i,t[i]);
    }
          });
      }
	
	TeamVectorQR_FormQ_Internal
	  ::invoke(member,
		   n, matrix_rank, matrix_rank,
		   A, as1, as0,
		   t, ts0,
		   V, vs1, vs0,
		   work);
      }
      if (do_print) {
          Kokkos::single(Kokkos::PerTeam(member), [&] () {
    printf("L2=zeros(%d,%d);\n", m, n);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<n; ++j) {
            printf("L2(%d,%d)= %f;\n", i+1,j+1,A[i*as0+j]);
        }
    }
    printf("V=zeros(%d,%d);\n", n, n);
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            printf("V(%d,%d)= %f;\n", i+1,j+1,V[i*vs0+j]);
        }
    }
          });
      }

      return 0;
    }
  };

} // end namespace KokkosBatched


#endif
