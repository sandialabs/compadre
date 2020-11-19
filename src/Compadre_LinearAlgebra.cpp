#include "Compadre_LinearAlgebra_Definitions.hpp"
#include "Compadre_Functors.hpp"
//#include <KokkosBlas.hpp>
//#include <KokkosBatched_LU_Decl.hpp>
//#include <KokkosBatched_LU_Serial_Impl.hpp>
//#include <KokkosBatched_LU_Team_Impl.hpp>
//#include <KokkosBatched_Trsv_Decl.hpp>
//#include <KokkosBatched_Trsv_Serial_Impl.hpp>
//#include <KokkosBatched_Trsv_Team_Impl.hpp>
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_ApplyPivot_Decl.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Trsv_Decl.hpp"
#include "KokkosBatched_UTV_Decl.hpp"
#include "KokkosBatched_SolveUTV_Decl.hpp"

using namespace KokkosBatched;

namespace Compadre{
namespace GMLS_LinearAlgebra {

  template<typename DeviceType,
           typename AlgoTagType,
           typename MatrixViewType,
           typename PivViewType,
           typename VectorViewType>
  struct Functor_TestBatchedTeamVectorSolveUTV {
    MatrixViewType _a;
    VectorViewType _b;

    int _pm_getTeamScratchLevel_1;
    int _M, _N, _NRHS;

    KOKKOS_INLINE_FUNCTION
    Functor_TestBatchedTeamVectorSolveUTV(
                      const int M,
                      const int N,
                      const int NRHS,
                      const MatrixViewType &a,
                      const VectorViewType &b)
      : _a(a), _b(b), _M(M), _N(N), _NRHS(NRHS) { _pm_getTeamScratchLevel_1 = 0; }

    template<typename MemberType>
    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &member) const {
      //typedef typename MatrixViewType::non_const_value_type value_type;
      //const value_type one(1), zero(0), add_this(10);

      const int k = member.league_rank();
      //auto rr = Kokkos::subview(_r, k, Kokkos::ALL(), Kokkos::ALL());
      //auto ac = Kokkos::subview(_acopy, k, Kokkos::ALL(), Kokkos::ALL());

      auto aaa = scratch_matrix_right_type(_a.data() + TO_GLOBAL(k)*TO_GLOBAL(_a.extent(1))*TO_GLOBAL(_a.extent(2)), _a.extent(1), _a.extent(2));
      scratch_matrix_right_type aa(member.team_scratch(_pm_getTeamScratchLevel_1), _M, _N);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,_M),[&](const int &i) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member,0,_N),[&](const int &j) {
            aa(i,j) = aaa(i,j);
        });
      });

      //auto aa = Kokkos::subview(_a, k, std::pair<int,int>(0,_M), std::pair<int,int>(0,_M));
      //auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
      auto bbb_right = scratch_matrix_right_type(_b.data() + TO_GLOBAL(k)*TO_GLOBAL(_b.extent(1))*TO_GLOBAL(_b.extent(2)), _b.extent(1), _b.extent(2));
      auto bbb_left = scratch_matrix_left_type(_b.data() + TO_GLOBAL(k)*TO_GLOBAL(_b.extent(1))*TO_GLOBAL(_b.extent(2)), _b.extent(1), _b.extent(2));
      scratch_matrix_right_type bb(member.team_scratch(_pm_getTeamScratchLevel_1), _M, _NRHS);

      if (std::is_same<typename decltype(_b)::array_layout, layout_left>::value) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,_M),[&](const int &i) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member,0,_N),[&](const int &j) {
              bb(i,j) = bbb_left(i,j);
          });
        });
      } else {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,_M),[&](const int &i) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(member,0,_NRHS),[&](const int &j) {
              bb(i,j) = bbb_right(i,j);
          });
        });
      }

      scratch_matrix_right_type uu(member.team_scratch(_pm_getTeamScratchLevel_1), _M, _M);
      scratch_matrix_right_type vv(member.team_scratch(_pm_getTeamScratchLevel_1), _N, _N);
      scratch_matrix_right_type xx(member.team_scratch(_pm_getTeamScratchLevel_1), _N, _NRHS);
      //for (int i=0; i<_N; ++i) {
      //  for (int j=0; j<_NRHS; ++j) {
      //      xx(i,j) = 0.0;
      //  }
      //}

      scratch_vector_type ww(member.team_scratch(_pm_getTeamScratchLevel_1), 3*_M*_NRHS);
      scratch_local_index_type pp(member.team_scratch(_pm_getTeamScratchLevel_1), _N);
      //for (int i=0; i<_N; ++i) {
      //  pp(i) = 0;
      //}

      //auto uu = Kokkos::subview(_u, k, Kokkos::ALL(), Kokkos::ALL());
      //auto vv = Kokkos::subview(_v, k, Kokkos::ALL(), Kokkos::ALL());
      //auto ww = Kokkos::subview(_w, k, Kokkos::ALL());

      //auto pp = Kokkos::subview(_p, k, Kokkos::ALL());
      //auto xx = Kokkos::subview(_x, k, Kokkos::ALL(), Kokkos::ALL());


      bool do_print = false;
      if (do_print) {
      //print a
    printf("a=zeros(%d,%d);\n", _a.extent(1), _a.extent(2));
    for (int i=0; i<_a.extent(0); ++i) {
        for (int j=0; j<_a.extent(1); ++j) {
            for (int k=0; k<_a.extent(2); ++k) {
                printf("a(%d,%d)= %f;\n", j+1,k+1, _a(i,j,k));
            }
        }
    }
      //print b
    printf("b=zeros(%d,%d);\n", _b.extent(1), _b.extent(2));
    for (int i=0; i<_b.extent(0); ++i) {
        for (int j=0; j<_b.extent(1); ++j) {
            for (int k=0; k<_b.extent(2); ++k) {
                printf("b(%d,%d)= %f;\n", j+1,k+1, _b(i,j,k));
            }
        }
    }
      }

    //  // make diagonal dominant and set xx = 1,2,3,4,5
    //  const int m = aa.extent(0), r = rr.extent(1);
    //  if (m <= r) {
    //Kokkos::parallel_for
    //  (Kokkos::TeamVectorRange(member, m),
    //   [&](const int &i) {
    //     aa(i,i) += add_this;
    //     xx(i) = (i+1);
    //   });
    //  } else {
    //Kokkos::parallel_for
    //  (Kokkos::TeamVectorRange(member, m*m),
    //   [&](const int &ij) {
    //        const int i = ij/m, j = ij%m;
    //        value_type tmp(0);
    //        for (int l=0;l<r;++l)
    //          tmp += rr(i,l)*rr(j,l);
    //        aa(i,j) = tmp;
    //      });
    //Kokkos::parallel_for
    //  (Kokkos::TeamVectorRange(member, m),
    //   [&](const int &i) {
    //        xx(i) = (i+1);
    //      });
    //  }

    //  /// copy for verification
    //  TeamVectorCopy<MemberType,Trans::NoTranspose>
    //::invoke(member, aa, ac);

    //  /// bb = AA*xx
    //  TeamVectorGemv<MemberType,Trans::NoTranspose,Algo::Gemv::Unblocked>
    //::invoke(member, one, aa, xx, zero, bb);
    //  member.team_barrier();

      /// Solving Ax = b using UTV transformation
      /// A P^T P x = b
      /// UTV P x = b;

      /// UTV = A P^T
      int matrix_rank(0);
      member.team_barrier();
      TeamVectorUTV<MemberType,AlgoTagType>
    ::invoke(member, aa, pp, uu, vv, ww, matrix_rank);
      member.team_barrier();
      if (do_print) printf("matrix_rank: %d\n", matrix_rank);

      if (do_print) {
      //print u
    printf("u=zeros(%d,%d);\n", uu.extent(0), uu.extent(1));
    //for (int i=0; i<_u.extent(0); ++i) {
        for (int j=0; j<uu.extent(0); ++j) {
            for (int k=0; k<uu.extent(1); ++k) {
                printf("u(%d,%d)= %f;\n", j+1,k+1, uu(j,k));
            }
        }
    //}
      }
    //  //print v
    //for (int i=0; i<_v.extent(0); ++i) {
    //    for (int j=0; j<_v.extent(1); ++j) {
    //        for (int k=0; k<_v.extent(2); ++k) {
    //            printf("v(%d,%d): %f\n", j,k, _v(i,j,k));
    //        }
    //    }
    //}
    //for (int i=0; i<_b.extent(0); ++i) {
    //    for (int j=0; j<_b.extent(1); ++j) {
    //        for (int k=0; k<_b.extent(2); ++k) {
    //            printf("b(%d,%d): %f\n", j,k, _b(i,j,k));
    //        }
    //    }
    //}
    //printf("p size: %d\n", _p.extent(1));
    //  if (do_print) {
    //printf("p=zeros(%d,1);\n", _p.extent(1));
    //for (int i=0; i<_p.extent(0); ++i) {
    //    for (int j=0; j<_p.extent(1); ++j) {
    //        printf("p(%d)= %d;\n", j+1, _p(i,j));
    //    }
    //}
    //  }
      TeamVectorSolveUTV<MemberType,AlgoTagType>
    ::invoke(member, matrix_rank, uu, aa, vv, pp, xx, bb, ww);
      member.team_barrier();
    //  if (do_print) {
    //for (int i=0; i<_x.extent(0); ++i) {
    //    for (int j=0; j<_x.extent(1); ++j) {
    //        for (int k=0; k<_x.extent(2); ++k) {
    //            printf("x(%d,%d): %f\n", j,k, _x(i,j,k));
    //        }
    //    }
    //}
    //  }

      Kokkos::parallel_for(Kokkos::TeamThreadRange(member,0,_N),[&](const int &i) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member,0,_NRHS),[&](const int &j) {
            bbb_right(i,j) = xx(i,j);
        });
      });

    }

    inline
    void run(ParallelManager pm) {
      typedef typename MatrixViewType::non_const_value_type value_type;
      std::string name_region("KokkosBatched::Test::TeamVectorSolveUTV");
      std::string name_value_type = ( std::is_same<value_type,float>::value ? "::Float" :
                                      std::is_same<value_type,double>::value ? "::Double" :
                                      std::is_same<value_type,Kokkos::complex<float> >::value ? "::ComplexFloat" :
                                      std::is_same<value_type,Kokkos::complex<double> >::value ? "::ComplexDouble" : "::UnknownValueType" );
      std::string name = name_region + name_value_type;
      Kokkos::Profiling::pushRegion( name.c_str() );

      //const int league_size = _a.extent(0);
      _pm_getTeamScratchLevel_1 = pm.getTeamScratchLevel(1);
      
      int scratch_size = scratch_matrix_right_type::shmem_size(_M, _M); // U
      scratch_size += scratch_matrix_right_type::shmem_size(_N, _N); // V
      scratch_size += scratch_vector_type::shmem_size(3*_M*_NRHS); // W
      scratch_size += scratch_matrix_right_type::shmem_size(_M,_N); // A (temporary)
      scratch_size += scratch_matrix_right_type::shmem_size(_M,_NRHS); // B (temporary)
      scratch_size += scratch_matrix_right_type::shmem_size(_N,_NRHS); // X (temporary)
      scratch_size += scratch_vector_type::shmem_size(_N); // P (temporary)

      pm.clearScratchSizes();
      pm.setTeamScratchSize(1, scratch_size);
      //pm.CallFunctorWithTeamThreads(_a.extent(0), *this);
      pm.CallFunctorWithTeamThreadsAndVectors(_a.extent(0), 1, 16, *this);

      Kokkos::fence();
      //Kokkos::TeamPolicy<DeviceType> policy(league_size, Kokkos::AUTO);
      //Kokkos::parallel_for(name.c_str(), policy, *this);

      Kokkos::Profiling::popRegion();
    }
  };



void batchQRFactorize(ParallelManager pm, double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, int NRHS, const int num_matrices, const size_t max_neighbors, const int initial_index_of_batch, int * neighbor_list_sizes) {



    typedef Algo::UTV::Unblocked algo_tag_type;
    //typedef Kokkos::View<double***, layout_right, host_execution_space>
    //                MatrixViewType;
    typedef Kokkos::View<double***, layout_right, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                    MatrixViewType;
    //MatrixViewType a("a", num_matrices, M, N);

    MatrixViewType A(P, num_matrices, lda, nda);
    //for (int i=0; i<num_matrices; ++i) {
    //    scratch_matrix_right_type PsqrtW(P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda), lda, nda);
    //    for (int j=0; j<M; ++j) {
    //        for (int k=0; k<N; ++k) {
    //            a(i,j,k) = PsqrtW(j,k);
    //        }
    //    }
    //}

    // can be moved to scratch
    //MatrixViewType u("u", num_matrices, M, M);
    //MatrixViewType v("v", num_matrices, N, N);

    typedef Kokkos::View<int**, layout_right, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                    PivViewType;
    // can be moved to scratch
    //PivViewType    p("p", num_matrices, N);

    typedef Kokkos::View<double***, layout_right, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                    VectorViewType;
    VectorViewType B(RHS, num_matrices, ldb, ndb);
    //VectorViewType x("x", num_matrices, N, NRHS);
    //VectorViewType b("b", num_matrices, M, NRHS);

    //for (int i=0; i<num_matrices; ++i) {
    //    scratch_matrix_right_type i_RHS(RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb), ldb, ndb);
    //    for (int j=0; j<M; ++j) {
    //        for (int k=0; k<NRHS; ++k) {
    //            b(i,j,k) = i_RHS(j,k);
    //        }
    //    }
    //}

    //typedef Kokkos::View<double**, layout_right, host_execution_space>
    //                WorkViewType;
    //WorkViewType   w("w", num_matrices, 3*M*NRHS);
    //Kokkos::fence();

    Functor_TestBatchedTeamVectorSolveUTV
      <device_execution_space, algo_tag_type, MatrixViewType, PivViewType, VectorViewType>(M,N,NRHS,A,B).run(pm);
    //Kokkos::fence();

    //for (int i=0; i<num_matrices; ++i) {
    //    scratch_matrix_right_type i_RHS(RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb), ldb, ndb);
    //    for (int j=0; j<N; ++j) {
    //        for (int k=0; k<NRHS; ++k) {
    //            i_RHS(j,k) = x(i,j,k);
    //        }
    //    }
    //}

////    // P was constructed layout right, while LAPACK and CUDA expect layout left
////    // P is not squared and not symmetric, so we must convert it to layout left
////    // RHS is symmetric and square, so no conversion is necessary
////    int scratch_size = scratch_matrix_left_type::shmem_size(lda, nda);
////    pm.clearScratchSizes();
////    pm.setTeamScratchSize(1, scratch_size);
////    ConvertLayoutRightToLeft crl(pm, lda, nda, P);
////    pm.CallFunctorWithTeamThreads(num_matrices, crl);
////    Kokkos::fence();
////
////#ifdef COMPADRE_USE_CUDA
////
////    Kokkos::Profiling::pushRegion("QR::Setup(Pointers)");
////    Kokkos::View<size_t*> array_P_RHS("P and RHS matrix pointers on device", 2*num_matrices);
////
////    // get pointers to device data
////    Kokkos::parallel_for(Kokkos::RangePolicy<device_execution_space>(0,num_matrices), KOKKOS_LAMBDA(const int i) {
////        array_P_RHS(i               ) = reinterpret_cast<size_t>(P   + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda));
////        array_P_RHS(i + num_matrices) = reinterpret_cast<size_t>(RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb));
////    });
////
////    Kokkos::View<int*> devInfo("devInfo", num_matrices);
////    cudaDeviceSynchronize();
////    Kokkos::Profiling::popRegion();
////
////    Kokkos::Profiling::pushRegion("QR::Setup(Handle)");
////    // Create cublas instance
////    cublasHandle_t cublas_handle;
////    cublasStatus_t cublas_stat;
////    cudaDeviceSynchronize();
////    Kokkos::Profiling::popRegion();
////
////    Kokkos::Profiling::pushRegion("QR::Setup(Create)");
////    cublasCreate(&cublas_handle);
////    cudaDeviceSynchronize();
////    Kokkos::Profiling::popRegion();
////
////    Kokkos::Profiling::pushRegion("QR::Execution");
////    int info;
////
////    // call batched QR
////    cublas_stat=cublasDgelsBatched(cublas_handle, CUBLAS_OP_N, 
////                                   M, N, NRHS,
////                                   reinterpret_cast<double**>(array_P_RHS.data()), lda,
////                                   reinterpret_cast<double**>(array_P_RHS.data() + TO_GLOBAL(num_matrices)), ldb,
////                                   &info, devInfo.data(), num_matrices );
////
////    compadre_assert_release(cublas_stat==CUBLAS_STATUS_SUCCESS && "cublasDgeqrfBatched failed");
////    cudaDeviceSynchronize();
////    Kokkos::Profiling::popRegion();
////
////
////#elif defined(COMPADRE_USE_LAPACK)
////
////    // requires column major input
////
////    Kokkos::Profiling::pushRegion("QR::Setup(Create)");
////
////    // find optimal blocksize when using LAPACK in order to allocate workspace needed
////    int lwork = -1, info = 0; double wkopt = 0;
////
////    if (num_matrices > 0) {
////        dgels_( (char *)"N", &M, &N, &NRHS, 
////                (double *)NULL, &lda, 
////                (double *)NULL, &ldb, 
////                &wkopt, &lwork, &info );
////    }
////
////    // size needed to malloc for each problem
////    lwork = (int)wkopt;
////
////    Kokkos::Profiling::popRegion();
////
////    std::string transpose_or_no = "N";
////
////    #ifdef LAPACK_DECLARED_THREADSAFE
////
////        int scratch_space_size = scratch_vector_type::shmem_size( lwork );  // work space
////
////        Kokkos::parallel_for(
////            team_policy(num_matrices, Kokkos::AUTO)
////            .set_scratch_size(0, Kokkos::PerTeam(scratch_space_size)),
////            KOKKOS_LAMBDA (const member_type& teamMember) {
////
////            scratch_vector_type scratch_work(teamMember.team_scratch(0), lwork);
////
////            int i_info = 0;
////        
////            const int i = teamMember.league_rank();
////
////            double * p_offset = P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda);
////            double * rhs_offset = RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb);
////
////            // use a custom # of neighbors for each problem, if possible
////            const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
////            int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i + initial_index_of_batch))*multiplier : M;
////            int my_num_rhs = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i + initial_index_of_batch))*multiplier : NRHS;
////
////            Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
////            dgels_( const_cast<char *>(transpose_or_no.c_str()), 
////                    const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rhs),
////                    p_offset, const_cast<int*>(&lda),
////                    rhs_offset, const_cast<int*>(&ldb),
////                    scratch_work.data(), const_cast<int*>(&lwork), &i_info);
////            });
////
////            compadre_assert_release(i_info==0 && "dgels failed");
////
////        }, "QR Execution");
////
////    #else
////
////        Kokkos::View<double*> scratch_work("scratch_work", lwork);
////        for (int i=0; i<num_matrices; ++i) {
////
////            int i_info = 0;
////        
////            double * p_offset = P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda);
////            double * rhs_offset = RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb);
////
////            // use a custom # of neighbors for each problem, if possible
////            const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
////            int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i + initial_index_of_batch))*multiplier : M;
////            int my_num_rhs = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i + initial_index_of_batch))*multiplier : NRHS;
////
////            dgels_( const_cast<char *>(transpose_or_no.c_str()), 
////                    const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rhs),
////                    p_offset, const_cast<int*>(&lda),
////                    rhs_offset, const_cast<int*>(&ldb),
////                    scratch_work.data(), const_cast<int*>(&lwork), &i_info);
////
////            compadre_assert_release(i_info==0 && "dgels failed");
////        }
////
////    #endif // LAPACK is not threadsafe
////
////#endif
////
////    // Results are written layout left, so they need converted to layout right
////    scratch_size = scratch_matrix_left_type::shmem_size(ldb, ndb);
////    pm.clearScratchSizes();
////    pm.setTeamScratchSize(1, scratch_size);
////    ConvertLayoutLeftToRight clr(pm, ldb, ndb, RHS);
////    pm.CallFunctorWithTeamThreads(num_matrices, clr);
////    Kokkos::fence();

}

void batchSVDFactorize(ParallelManager pm, bool swap_layout_P, double *P, int lda, int nda, bool swap_layout_RHS, double *RHS, int ldb, int ndb, int M, int N, int NRHS, const int num_matrices, const size_t max_neighbors, const int initial_index_of_batch, int * neighbor_list_sizes) {

    if (swap_layout_P) {
        // P was constructed layout right, while LAPACK and CUDA expect layout left
        // P is not squared and not symmetric, so we must convert it to layout left
        // RHS is symmetric and square, so no conversion is necessary
        int scratch_size = scratch_matrix_left_type::shmem_size(lda, nda);
        pm.clearScratchSizes();
        pm.setTeamScratchSize(1, scratch_size);
        ConvertLayoutRightToLeft crl(pm, lda, nda, P);
        pm.CallFunctorWithTeamThreads(num_matrices, crl);
        Kokkos::fence();
    }

#ifdef COMPADRE_USE_CUDA

    const int NUM_STREAMS = 64;

    Kokkos::Profiling::pushRegion("SVD::Setup(Allocate)");

      int ldu = M;
      int ldv = N;
      int min_mn = (M<N) ? M : N;
      int max_mn = (M>N) ? M : N;
      // local U, S, and V must be allocated
      Kokkos::View<double*> U("U", TO_GLOBAL(num_matrices)*TO_GLOBAL(ldu)*TO_GLOBAL(M));
      Kokkos::View<double*> V("V", TO_GLOBAL(num_matrices)*TO_GLOBAL(ldv)*TO_GLOBAL(N));
      Kokkos::View<double*> S("S", TO_GLOBAL(num_matrices)*TO_GLOBAL(min_mn));
      Kokkos::View<int*> devInfo("device info", num_matrices);

    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("SVD::Setup(Handle)");
      cudaError_t cudaStat1 = cudaSuccess;
      std::vector<cusolverDnHandle_t> cusolver_handles(NUM_STREAMS);
      cusolverStatus_t cusolver_stat = CUSOLVER_STATUS_SUCCESS;
      std::vector<cudaStream_t> streams(NUM_STREAMS);
      gesvdjInfo_t gesvdj_params = NULL;
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("SVD::Setup(Create)");

      for (int i=0; i<NUM_STREAMS; ++i) {
          cusolver_stat = cusolverDnCreate(&cusolver_handles[i]);
          compadre_assert_release(CUSOLVER_STATUS_SUCCESS == cusolver_stat && "DN Create");
          cudaStat1 = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
          compadre_assert_release(cudaSuccess == cudaStat1 && "Cuda Stream Create");
          cusolver_stat = cusolverDnSetStream(cusolver_handles[i], streams[i]);
          compadre_assert_release(CUSOLVER_STATUS_SUCCESS == cusolver_stat && "Cuda Set Stream");
      }


      cusolver_stat = cusolverDnCreateGesvdjInfo(&gesvdj_params);
      compadre_assert_release(CUSOLVER_STATUS_SUCCESS == cusolver_stat && "Create GesvdjInfo");

      const double tol = 1.e-14;
      const int max_sweeps = 15;
      const int sort_svd  = 1;   /* sort singular values */

      cusolver_stat = cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);
      compadre_assert_release(CUSOLVER_STATUS_SUCCESS == cusolver_stat && "Set Tolerance");

      cusolver_stat = cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);
      compadre_assert_release(CUSOLVER_STATUS_SUCCESS == cusolver_stat && "Set Sweeps");

      cusolver_stat = cusolverDnXgesvdjSetSortEig(gesvdj_params, sort_svd);
      compadre_assert_release(CUSOLVER_STATUS_SUCCESS == cusolver_stat && "Sort SVD");

      const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */

    Kokkos::Profiling::popRegion();

    if (max_mn <= 32) { // batch version only works on small matrices (https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvdjbatch)

        Kokkos::Profiling::pushRegion("SVD::Workspace");

          int lwork = 0;

          cusolver_stat = cusolverDnDgesvdjBatched_bufferSize(
              cusolver_handles[0], jobz,
              M, N,
              P, lda, // P
              S.data(), // S
              U.data(), ldu, // U
              V.data(), ldv, // V
              &lwork, gesvdj_params, num_matrices
          );
          compadre_assert_release(CUSOLVER_STATUS_SUCCESS == cusolver_stat && "Get Work Size");

          Kokkos::View<double*> work("work for cusolver", lwork);
          cudaStat1 = cudaDeviceSynchronize();
          compadre_assert_release(cudaSuccess == cudaStat1 && "Device Sync 1");
        Kokkos::Profiling::popRegion();

        Kokkos::Profiling::pushRegion("SVD::Execution");
          cusolver_stat = cusolverDnDgesvdjBatched(
              cusolver_handles[0], jobz,
              M, N,
              P, lda,
              S.data(),
              U.data(), ldu,
              V.data(), ldv,
              work.data(), lwork,
              devInfo.data(), gesvdj_params, num_matrices
          );
          cudaStat1 = cudaDeviceSynchronize();
          compadre_assert_release(CUSOLVER_STATUS_SUCCESS == cusolver_stat && "Solver Didn't Succeed");
          compadre_assert_release(cudaSuccess == cudaStat1 && "Device Sync 2");
        Kokkos::Profiling::popRegion();

    } else { // bigger than 32 (batched dgesvdj can only handle 32x32)
  
  
        Kokkos::Profiling::pushRegion("SVD::Workspace");
  
          int lwork = 0;
          int econ = 1;
  
          //cusolver_stat = cusolverDnDgesvd_bufferSize(
          //    cusolver_handles[0], M, N, &lwork );
          //compadre_assert_release (cusolver_stat == CUSOLVER_STATUS_SUCCESS && "dgesvd Work Size Failed");
  
          cusolver_stat = cusolverDnDgesvdj_bufferSize(
              cusolver_handles[0], jobz, econ,
              M, N,
              P, lda, // P
              S.data(), // S
              U.data(), ldu, // U
              V.data(), ldv, // V
              &lwork, gesvdj_params
          );
          compadre_assert_release(CUSOLVER_STATUS_SUCCESS == cusolver_stat && "Get Work Size");
  
          Kokkos::View<double*> work("CUDA work space", lwork*num_matrices);
          //Kokkos::View<double*> rwork("CUDA work space", (min_mn-1)*num_matrices);
          cudaDeviceSynchronize();
        Kokkos::Profiling::popRegion();
        
        //signed char jobu = 'A';
        //signed char jobv = 'A';
  
        Kokkos::Profiling::pushRegion("SVD::Execution");
          for (int i=0; i<num_matrices; ++i) {
              const int my_stream = i%NUM_STREAMS;

              cusolverDnDgesvdj(
                  cusolver_handles[my_stream], jobz, econ,
                  M, N,
                  P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda), lda,
                  S.data() + TO_GLOBAL(i)*TO_GLOBAL(min_mn),
                  U.data() + TO_GLOBAL(i)*TO_GLOBAL(ldu)*TO_GLOBAL(M), ldu,
                  V.data() + TO_GLOBAL(i)*TO_GLOBAL(ldv)*TO_GLOBAL(N), ldv,
                  work.data() + TO_GLOBAL(i)*TO_GLOBAL(lwork), lwork,
                  devInfo.data() + TO_GLOBAL(i), gesvdj_params
              );
  
//              cusolverDnDgesvd (
//                  cusolver_handles[my_stream],
//                  jobu,
//                  jobv,
//                  M,
//                  N,
//                  P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda),
//                  lda,
//                  S.data() + TO_GLOBAL(i)*TO_GLOBAL(min_mn),
//                  U.data() + TO_GLOBAL(i)*TO_GLOBAL(ldu)*TO_GLOBAL(M),
//                  ldu,
//                  V.data() + TO_GLOBAL(i)*TO_GLOBAL(ldv)*TO_GLOBAL(N),
//                  ldv,
//                  work.data() + TO_GLOBAL(i)*TO_GLOBAL(lwork),
//                  lwork,
//                  rwork.data() + TO_GLOBAL(i)*TO_GLOBAL((min_mn-1)),
//                  devInfo.data() + TO_GLOBAL(i) );
  
          }
        Kokkos::Profiling::popRegion();
    }

    cudaDeviceSynchronize();
    for (int i=0; i<NUM_STREAMS; ++i) {
        cusolverDnDestroy(cusolver_handles[i]);
    }


    Kokkos::Profiling::pushRegion("SVD::S Copy");
    // deep copy neighbor list sizes over to gpu
    Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_neighbor_list_sizes(neighbor_list_sizes + initial_index_of_batch, num_matrices);
    Kokkos::View<int*> d_neighbor_list_sizes("neighbor list sizes on device", num_matrices);
    Kokkos::deep_copy(d_neighbor_list_sizes, h_neighbor_list_sizes);

    int scratch_space_level;
    int scratch_space_size = 0;
    scratch_space_size += scratch_vector_type::shmem_size( min_mn );  // s
    if (swap_layout_P) {
        // Less memory is required if the right hand side matrix is diagonal
        scratch_space_level = 0;
    } else {
        // More memory is required to perform full matrix multiplication
        scratch_space_size += scratch_matrix_left_type::shmem_size(N, M);
        scratch_space_size += scratch_matrix_left_type::shmem_size(N, NRHS);
        scratch_space_level = 1;
    }
    Kokkos::Profiling::popRegion();
    
    Kokkos::parallel_for(
        team_policy(num_matrices, Kokkos::AUTO)
        .set_scratch_size(scratch_space_level, Kokkos::PerTeam(scratch_space_size)), // shared memory
        KOKKOS_LAMBDA (const member_type& teamMember) {

        const int target_index = teamMember.league_rank();

        // use a custom # of neighbors for each problem, if possible
        const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
        int my_num_rows = d_neighbor_list_sizes(target_index)*multiplier;
        //int my_num_rows = d_neighbor_list_sizes(target_index)*multiplier : M;

        scratch_vector_type s(teamMember.team_scratch(scratch_space_level), min_mn ); // shared memory

        // data is actually layout left
        scratch_matrix_left_type
            RHS_(RHS + TO_GLOBAL(target_index)*TO_GLOBAL(ldb)*TO_GLOBAL(NRHS), ldb, NRHS);
        scratch_matrix_left_type
            U_(U.data() + TO_GLOBAL(target_index)*TO_GLOBAL(ldu)*TO_GLOBAL(M), ldu, M);
        scratch_matrix_left_type
            V_(V.data() + TO_GLOBAL(target_index)*TO_GLOBAL(ldv)*TO_GLOBAL(N), ldv, N);
        scratch_vector_type
            S_(S.data() + TO_GLOBAL(target_index)*TO_GLOBAL(min_mn), min_mn);

        // threshold for dropping singular values
        double S0 = S_(0);
        double eps = 1e-14;
        double abs_threshold = eps*S0;        

        // t1 takes on the role of S inverse
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, min_mn), [=] (const int i) {
            s(i) = ( std::abs(S_(i)) > abs_threshold ) ? 1./S_(i) : 0;
        });
        teamMember.team_barrier();

        if (swap_layout_P) {
            // When solving PsqrtW against sqrtW, since there are two
            // diagonal matrices (s and the right hand side), the matrix
            // multiplication can be written more efficiently
            for (int i=0; i<my_num_rows; ++i) {
                double sqrt_w_i_m = RHS_(i,i);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,N),
                      [=] (const int j) {

                    double  bdataj = 0;
                    for (int k=0; k<min_mn; ++k) {
                        bdataj += V_(j,k)*s(k)*U_(i,k);
                    }
                // RHS_ is where we can find std::sqrt(w)
                    bdataj *= sqrt_w_i_m;
                    RHS_(j,i) = bdataj;
                });
                teamMember.team_barrier();
            }
        } else {
            // Otherwise, you need to perform a full matrix multiplication
            scratch_matrix_left_type temp_VSU_left_matrix(teamMember.team_scratch(scratch_space_level), N, M);
            scratch_matrix_left_type temp_x_left_matrix(teamMember.team_scratch(scratch_space_level), N, NRHS);
            // Multiply V s U^T and sotre into temp_VSU_left_matrix
            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, N),
                [=] (const int i) {
                for (int j=0; j<M; j++) {
                    double temp = 0.0;
                    for (int k=0; k<min_mn; k++) {
                        temp += V_(i,k)*s(k)*U_(j,k);
                    }
                    temp_VSU_left_matrix(i, j) = temp;
                }
            });
            teamMember.team_barrier();

            // Multiply temp_VSU_left_matrix with RHS_ and store in temp_x_left_matrix
            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, N),
                [=] (const int i) {
                for (int j=0; j<NRHS; j++) {
                    double temp = 0.0;
                    for (int k=0; k<M; k++) {
                        temp += temp_VSU_left_matrix(i, k)*RHS_(k, j);
                    }
                    temp_x_left_matrix(i, j) = temp;
                }
            });
            teamMember.team_barrier();
            // Copy the matrix back to RHS
            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, N),
                [=] (const int i) {
                for (int j=0; j<NRHS; j++) {
                    RHS_(i, j) = temp_x_left_matrix(i, j);
                }
            });
        }
    }, "SVD: USV*RHS Multiply");





#elif defined(COMPADRE_USE_LAPACK)

    // later improvement could be to send in an optional view with the neighbor list size for each target to reduce work

    Kokkos::Profiling::pushRegion("SVD::Setup(Create)");
    double rcond = 1e-14; // rcond*S(0) is cutoff for singular values in dgelsd

    const int smlsiz = 25;
    const int nlvl = std::max(0, int( std::log2( std::min(M,N) / (smlsiz+1) ) + 1));
    const int liwork = 3*std::min(M,N)*nlvl + 11*std::min(M,N);

    std::vector<int> iwork(liwork);
    std::vector<double> s(std::max(M,N));

    int lwork = -1;
    double wkopt = 0;
    int rank = 0;
    int info = 0;

    if (num_matrices > 0) {
        dgelsd_( &M, &N, &NRHS, 
                 P, &lda, 
                 RHS, &ldb, 
                 s.data(), &rcond, &rank,
                 &wkopt, &lwork, iwork.data(), &info);
    }
    lwork = (int)wkopt;

    Kokkos::Profiling::popRegion();

    #ifdef LAPACK_DECLARED_THREADSAFE
    
        int scratch_space_size = 0;
        scratch_space_size += scratch_vector_type::shmem_size( lwork );  // work space
        scratch_space_size += scratch_vector_type::shmem_size( std::max(M,N) );  // s
        scratch_space_size += scratch_local_index_type::shmem_size( liwork ); // iwork space
        
        Kokkos::parallel_for(
            team_policy(num_matrices, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_space_size)),
            KOKKOS_LAMBDA (const member_type& teamMember) {

                scratch_vector_type scratch_work(teamMember.team_scratch(0), lwork);
                scratch_vector_type scratch_s(teamMember.team_scratch(0), std::max(M,N) );
                scratch_local_index_type scratch_iwork(teamMember.team_scratch(0), liwork);

                int i_rank = 0;
                int i_info = 0;
        
                const int i = teamMember.league_rank();

                double * p_offset = P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda);
                double * rhs_offset = RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb);

                // use a custom # of neighbors for each problem, if possible
                const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
                int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i + initial_index_of_batch))*multiplier : M;
                int my_num_rhs = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i + initial_index_of_batch))*multiplier : NRHS;

                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                dgelsd_( const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rhs),
                         p_offset, const_cast<int*>(&lda),
                         rhs_offset, const_cast<int*>(&ldb),
                         scratch_s.data(), const_cast<double*>(&rcond), &i_rank,
                         scratch_work.data(), const_cast<int*>(&lwork), scratch_iwork.data(), &i_info);
                });

                compadre_assert_release(i_info==0 && "dgelsd failed");

        }, "SVD Execution");

    #else
    
        std::vector<double> scratch_work(lwork);
        std::vector<double> scratch_s(std::max(M,N));
        std::vector<int>    scratch_iwork(liwork);
        
        for (int i=0; i<num_matrices; ++i) {

                int i_rank = 0;
                int i_info = 0;
        
                double * p_offset = P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda);
                double * rhs_offset = RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb);

                // use a custom # of neighbors for each problem, if possible
                const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
                int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i + initial_index_of_batch))*multiplier : M;
                int my_num_rhs = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i + initial_index_of_batch))*multiplier : NRHS;

                dgelsd_( const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rhs),
                         p_offset, const_cast<int*>(&lda),
                         rhs_offset, const_cast<int*>(&ldb),
                         scratch_s.data(), const_cast<double*>(&rcond), &i_rank,
                         scratch_work.data(), const_cast<int*>(&lwork), scratch_iwork.data(), &i_info);

                compadre_assert_release(i_info==0 && "dgelsd failed");

        }

    #endif // LAPACK is not threadsafe

#endif

    // Results are written layout left, so they need converted to layout right
    if (swap_layout_RHS) {
        int scratch_size = scratch_matrix_left_type::shmem_size(ldb, ndb);
        pm.clearScratchSizes();
        pm.setTeamScratchSize(1, scratch_size);
        ConvertLayoutLeftToRight clr(pm, ldb, ndb, RHS);
        pm.CallFunctorWithTeamThreads(num_matrices, clr);
    }
    Kokkos::fence();

}

void batchLUFactorize(ParallelManager pm, double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, int NRHS, const int num_matrices, const size_t max_neighbors, const int initial_index_of_batch, int * neighbor_list_sizes) {

    int zzz = 0;
    int yyy = 0;
    //int scratch_size = scratch_matrix_left_type::shmem_size(ldb, ndb);
    //pm.clearScratchSizes();
    //pm.setTeamScratchSize(1, scratch_size);
    //ConvertLayoutLeftToRight clr(pm, ldb, ndb, RHS);
    //pm.CallFunctorWithTeamThreads(num_matrices, clr);
    //Kokkos::fence();

    printf("lda: %d, nda %d, ldb %d, ndb %d, M %d, N %d, NRHS %d\n", lda, nda, ldb, ndb, M, N, NRHS);

    //// look at non-square entries in P
    //for (int i=0; i<num_matrices; ++i) {
    //    scratch_matrix_right_type PsqrtW(P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda), lda, nda);
    //    for (int j=0; j<lda; ++j) {
    //        for (int k=0; k<nda; ++k) {
    //            if ((j>=M || k>=N)  && PsqrtW(j,k) != 0.0) {
    //                printf("a(%d,%d,%d) = %f\n", i, j, k, PsqrtW(j,k));
    //            } else if (PsqrtW(j,k)!=PsqrtW(k,j)) {
    //                printf("NOT SYMMETRIC!! b(%d,%d,%d) = %f\n", i, j, k, PsqrtW(j,k));
    //            }
    //        }
    //    }
    //}

//    int mNRHS=2;
//    typedef Algo::UTV::Unblocked algo_tag_type;
//    typedef Kokkos::View<double***, layout_right, host_execution_space>
//                    MatrixViewType;
//    MatrixViewType a("a", 1, 4, 2);
//    a(0,0,0)=1.0;
//    a(0,1,0)=2.0;
//    a(0,2,0)=3.0;
//    a(0,3,0)=4.0;
//    MatrixViewType u("u", 1, 4, 4);
//    MatrixViewType v("v", 1, 2, 2);
//    typedef Kokkos::View<int**, layout_right, host_execution_space>
//                    PivViewType;
//    // can be moved to scratch
//    PivViewType    p("p", 1, 2);
//    typedef Kokkos::View<double***, layout_right, host_execution_space>
//                    VectorViewType;
//    MatrixViewType x("x", 1, 2, mNRHS);
//    MatrixViewType b("b", 1, 4, mNRHS);
//    b(0,0,0) = 1.0;
//    b(0,1,0) = 1.0;
//    b(0,2,0) = 1.0;
//    b(0,3,0) = 1.0;
//    b(0,0,1) = 1.0;
//    b(0,1,1) = 2.0;
//    b(0,2,1) = 3.0;
//    b(0,3,1) = 4.0;
//    typedef Kokkos::View<double**, layout_right, host_execution_space>
//                    WorkViewType;
//    WorkViewType   w("w", 1, 3*4*2*mNRHS);
//    Kokkos::fence();
//    Functor_TestBatchedTeamVectorSolveUTV
//      <device_execution_space,MatrixViewType,VectorViewType,
//       PivViewType,WorkViewType,algo_tag_type>(a,u,v,p,x,b,w).run();
//    printf("here a!!!\n");
//    Kokkos::fence();

    //Kokkos::fence();
    typedef Algo::UTV::Unblocked algo_tag_type;
    //typedef Kokkos::View<double***, layout_right, host_execution_space>
    //                MatrixViewType;
    typedef Kokkos::View<double***, layout_right, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                    MatrixViewType;
    //MatrixViewType a("a", num_matrices, M+zzz, N+yyy);

    MatrixViewType A(P, num_matrices, lda, nda);
    //for (int i=0; i<num_matrices; ++i) {
    //    scratch_matrix_right_type PsqrtW(P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda), lda, nda);
    //    for (int j=0; j<M; ++j) {
    //        for (int k=0; k<N; ++k) {
    //            a(i,j,k) = PsqrtW(j,k);
    //        }
    //    }
    //}

    //// can be moved to scratch
    //MatrixViewType u("u", num_matrices, M+zzz, M+zzz);
    //MatrixViewType v("v", num_matrices, N+yyy, N+yyy);

    typedef Kokkos::View<int**, layout_right, host_execution_space>
                    PivViewType;
    // can be moved to scratch
    //PivViewType    p("p", num_matrices, N+yyy);

    typedef Kokkos::View<double***, layout_left, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
                    VectorViewType;
    VectorViewType B(RHS, num_matrices, ldb, ndb);
    //typedef Kokkos::View<double***, layout_right, host_execution_space>
    //                VectorViewType;
    //VectorViewType x("x", num_matrices, N+yyy, NRHS);
    //VectorViewType b("b", num_matrices, M+zzz, NRHS);

    //for (int i=0; i<num_matrices; ++i) {
    //    //scratch_matrix_right_type i_RHS(RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb), M, NRHS);
    //    scratch_matrix_left_type i_RHS(RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb), ldb, ndb);
    //    for (int j=0; j<M; ++j) {
    //        for (int k=0; k<NRHS; ++k) {
    //            b(i,j,k) = i_RHS(j,k);
    //        }
    //    }
    //}
    //Kokkos::fence();

    //typedef Kokkos::View<double**, layout_right, host_execution_space>
    //                WorkViewType;
    //WorkViewType   w("w", num_matrices, 3*(M+zzz)*NRHS);
    //Kokkos::fence();

    Functor_TestBatchedTeamVectorSolveUTV
      <device_execution_space, algo_tag_type, MatrixViewType, PivViewType, VectorViewType>(M,N,NRHS,A,B).run(pm);

    //printf("here a!!!\n");
    //Kokkos::fence();
    ////printf("here b!!!\n");

    //for (int i=0; i<num_matrices; ++i) {
    //    scratch_matrix_right_type i_RHS(RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb), ldb, ndb);
    //    //for (int j=0; j<NRHS*M; ++j) {
    //    //    printf("QR PIVOT: %d, %d, %d, %f\n", i,j,*(x.data()+j));
    //    //    *(RHS+j) = *(x.data()+j);
    //    //}

    //    for (int j=0; j<N; ++j) {
    //        for (int k=0; k<NRHS; ++k) {
    //            i_RHS(j,k) = x(i,j,k);
    //            //printf("x(%d,%d,%d) = %f\n", i, j, k, x(i,j,k));
    //        }
    //    }
    //}
    //Kokkos::fence();

    // P was constructed layout right, while LAPACK and CUDA expect layout left
    // P is squared and symmetric so no layout conversion necessary
    // RHS is not square and not symmetric. However, the implicit cast to layout left
    // is equivalent to transposing the matrix and being consistent with layout left
    // Essentially, two operations for free.
    

    ////swap back
    //scratch_size = scratch_matrix_left_type::shmem_size(ldb, ndb);
    //pm.clearScratchSizes();
    //pm.setTeamScratchSize(1, scratch_size);
    //ConvertLayoutRightToLeft crl(pm, ldb, ndb, RHS);
    //pm.CallFunctorWithTeamThreads(num_matrices, crl);
    //Kokkos::fence();

////#ifdef COMPADRE_USE_CUDA
////
////    Kokkos::Profiling::pushRegion("LU::Setup(Pointers)");
////    Kokkos::View<size_t*> array_P_RHS("P and RHS matrix pointers on device", 2*num_matrices);
////    Kokkos::View<int*> ipiv_device("ipiv space on device", num_matrices*M);
////
////    // get pointers to device data
////    Kokkos::parallel_for(Kokkos::RangePolicy<device_execution_space>(0,num_matrices), KOKKOS_LAMBDA(const int i) {
////        array_P_RHS(i               ) = reinterpret_cast<size_t>(P   + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda));
////        array_P_RHS(i + num_matrices) = reinterpret_cast<size_t>(RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb));
////    });
////
////    Kokkos::View<int*> devInfo("devInfo", num_matrices);
////    cudaDeviceSynchronize();
////    Kokkos::Profiling::popRegion();
////
////    Kokkos::Profiling::pushRegion("LU::Setup(Handle)");
////    // Create cublas instance
////    cublasHandle_t cublas_handle;
////    cublasStatus_t cublas_stat;
////    cudaDeviceSynchronize();
////    Kokkos::Profiling::popRegion();
////
////    Kokkos::Profiling::pushRegion("LU::Setup(Create)");
////    cublasCreate(&cublas_handle);
////    cudaDeviceSynchronize();
////    Kokkos::Profiling::popRegion();
////
////    int info = 0;
////
////    Kokkos::Profiling::pushRegion("LU::Execution");
////
////    // call batched LU factorization
////    cublas_stat=cublasDgetrfBatched(cublas_handle, 
////                                   M,
////                                   reinterpret_cast<double**>(array_P_RHS.data()), lda,
////                                   reinterpret_cast<int*>(ipiv_device.data()),
////                                   devInfo.data(), num_matrices );
////    compadre_assert_release(cublas_stat==CUBLAS_STATUS_SUCCESS && "cublasDgetrfBatched failed");
////
////    // call batched LU application
////    cublas_stat=cublasDgetrsBatched(cublas_handle, CUBLAS_OP_N,
////                                   M, NRHS,
////                                   reinterpret_cast<const double**>(array_P_RHS.data()), lda,
////                                   reinterpret_cast<int*>(ipiv_device.data()),
////                                   reinterpret_cast<double**>(array_P_RHS.data() + TO_GLOBAL(num_matrices)), ldb,
////                                   &info, num_matrices );
////    compadre_assert_release(cublas_stat==CUBLAS_STATUS_SUCCESS && "cublasDgetrsBatched failed");
////
////    cudaDeviceSynchronize();
////    Kokkos::Profiling::popRegion();
////
////
////#elif defined(COMPADRE_USE_LAPACK)
////
//////    Kokkos::View<double***> AA("AA", num_matrices, M, N); /// element matrices
//////    Kokkos::View<double**>  BB("BB", num_matrices, N, N);    /// load vector and would be overwritten by a solution
//////    
//////    using namespace KokkosBatched;
//////#if 0 // range policy
//////    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
//////      auto A = Kokkos::subview(AA, i, Kokkos::ALL(), Kokkos::ALL()); /// ith matrix
//////      auto B = Kokkos::subview(BB, i, Kokkos::ALL());                /// ith load/solution vector
//////    
//////      SerialLU<Algo::LU::Unblocked>
//////        ::invoke(A);
//////      SerialTrsv<Uplo::Lower,Trans::NoTranspose,Diag::Unit,Algo::Trsv::Unblocked>
//////        ::invoke(1, A, B);
//////    });
//////#endif
//////
//////#if 1 // team policy
//////    {
//////      typedef Kokkos::TeamPolicy<device_execution_space> team_policy_type;
//////      typedef typename team_policy_type::member_type member_type;
//////      const int team_size = 32, vector_size = 1;
//////      team_policy_type policy(num_matrices, team_size, vector_size); 
//////      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const member_type &member) {
//////	  const int i = member.league_rank();
//////	    auto A = Kokkos::subview(AA, i, Kokkos::ALL(), Kokkos::ALL()); /// ith matrix
//////	    auto B = Kokkos::subview(BB, i, Kokkos::ALL());                /// ith load/solution vector
//////	    
//////	    TeamLU<member_type,Algo::LU::Unblocked>
//////	      ::invoke(member, A);
//////	    TeamTrsv<member_type,Uplo::Lower,Trans::NoTranspose,Diag::Unit,Algo::Trsv::Unblocked>
//////	      ::invoke(member, 1, A, B);
//////	  });
//////	}
//////#endif
//////
//////#endif
////
////    // later improvement could be to send in an optional view with the neighbor list size for each target to reduce work
////
////    Kokkos::Profiling::pushRegion("LU::Setup(Create)");
////    Kokkos::Profiling::popRegion();
////
////    std::string transpose_or_no = "N";
////
////    #ifdef LAPACK_DECLARED_THREADSAFE
////        int scratch_space_size = 0;
////        scratch_space_size += scratch_local_index_type::shmem_size( std::min(M, N) );  // ipiv
////
////        Kokkos::parallel_for(
////            team_policy(num_matrices, Kokkos::AUTO)
////            .set_scratch_size(0, Kokkos::PerTeam(scratch_space_size)),
////            KOKKOS_LAMBDA (const member_type& teamMember) {
////
////                scratch_local_index_type scratch_ipiv(teamMember.team_scratch(0), std::min(M, N));
////                int i_info = 0;
////
////                const int i = teamMember.league_rank();
////                double * p_offset = P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda);
////                double * rhs_offset = RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb);
////
////                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
////                dgetrf_( const_cast<int*>(&M), const_cast<int*>(&N),
////                         p_offset, const_cast<int*>(&lda),
////                         scratch_ipiv.data(),
////                         &i_info);
////                });
////                teamMember.team_barrier();
////
////                compadre_assert_release(i_info==0 && "dgetrf failed");
////
////                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
////                dgetrs_( const_cast<char *>(transpose_or_no.c_str()),
////                         const_cast<int*>(&N), const_cast<int*>(&NRHS),
////                         p_offset, const_cast<int*>(&lda),
////                         scratch_ipiv.data(),
////                         rhs_offset, const_cast<int*>(&ldb),
////                         &i_info);
////                });
////
////                compadre_assert_release(i_info==0 && "dgetrs failed");
////
////        }, "LU Execution");
////
////    #else
////
////        Kokkos::View<int*> scratch_ipiv("scratch_ipiv", std::min(M, N));
////
////        for (int i=0; i<num_matrices; ++i) {
////
////                int i_info = 0;
////
////                double * p_offset = P + TO_GLOBAL(i)*TO_GLOBAL(lda)*TO_GLOBAL(nda);
////                double * rhs_offset = RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb);
////
////                dgetrf_( const_cast<int*>(&M), const_cast<int*>(&N),
////                         p_offset, const_cast<int*>(&lda),
////                         scratch_ipiv.data(),
////                         &i_info);
////
////                dgetrs_( const_cast<char *>(transpose_or_no.c_str()),
////                         const_cast<int*>(&N), const_cast<int*>(&NRHS),
////                         p_offset, const_cast<int*>(&lda),
////                         scratch_ipiv.data(),
////                         rhs_offset, const_cast<int*>(&ldb),
////                         &i_info);
////
////                compadre_assert_release(i_info==0 && "dgetrs failed");
////
////        }
////
////    #endif // LAPACK is not threadsafe
////
////#endif
////
////    //// Results are written layout left, so they need converted to layout right
////    int scratch_size = scratch_matrix_left_type::shmem_size(ldb, ndb);
////    pm.clearScratchSizes();
////    pm.setTeamScratchSize(1, scratch_size);
////    ConvertLayoutLeftToRight clr2(pm, ldb, ndb, RHS);
////    pm.CallFunctorWithTeamThreads(num_matrices, clr2);
////    Kokkos::fence();
////    for (int i=0; i<num_matrices; ++i) {
////        scratch_matrix_right_type i_RHS(RHS + TO_GLOBAL(i)*TO_GLOBAL(ldb)*TO_GLOBAL(ndb), ldb, ndb);
////        for (int j=0; j<M; ++j) {
////            for (int k=0; k<NRHS; ++k) {
////                //i_RHS(j,k) = x(i,j,k);
////                //printf("LU: %d, %d, %d, %f\n", i,j,*(RHS+j));
////                if (std::abs(i_RHS(j,k)-x(i,j,k))>1e-5) {
////                    printf("LU: %d, %d, %d, %f vs %f\n", i,j,k,i_RHS(j,k),x(i,j,k));
////                }
////            }
////        //for (int j=0; j<N; ++j) {
////        //    for (int k=0; k<NRHS; ++k) {
////        //        //i_RHS(j,k) = x(i,j,k);
////        //        printf("LU: %d, %d, %d, %f\n", i,j,k,i_RHS(j,k));
////        //    }
////        }
////    }

}

} // GMLS_LinearAlgebra
} // Compadre
