#include "Compadre_LinearAlgebra_Definitions.hpp"

namespace Compadre{
namespace GMLS_LinearAlgebra {

void batchQRFactorize(double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, const int num_matrices, const size_t max_neighbors, int * neighbor_list_sizes) {

#ifdef COMPADRE_USE_CUDA

    Kokkos::Profiling::pushRegion("QR::Setup(Pointers)");
    Kokkos::View<size_t*> array_P_RHS("P and RHS matrix pointers on device", 2*num_matrices);

    // get pointers to device data
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA(const int i) {
        array_P_RHS(i               ) = reinterpret_cast<size_t>(P   + i*lda*nda);
        array_P_RHS(i + num_matrices) = reinterpret_cast<size_t>(RHS + i*ldb*ndb);
    });

    int *devInfo; cudaMalloc(&devInfo, num_matrices*sizeof(int));
    cudaDeviceSynchronize();
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("QR::Setup(Handle)");
    // Create cublas instance
    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_stat;
    cudaDeviceSynchronize();
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("QR::Setup(Create)");
    cublasCreate(&cublas_handle);
    cudaDeviceSynchronize();
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("QR::Execution");
    int info;

    // call batched QR
    cublas_stat=cublasDgelsBatched(cublas_handle, CUBLAS_OP_N, 
                                   M, N, M,
                                   reinterpret_cast<double**>(array_P_RHS.ptr_on_device()), lda,
                                   reinterpret_cast<double**>(array_P_RHS.ptr_on_device() + num_matrices), ldb,
                                   &info, devInfo, num_matrices );

    if (cublas_stat != CUBLAS_STATUS_SUCCESS)
      printf("\n cublasDgeqrfBatched failed");
    cudaDeviceSynchronize();
    Kokkos::Profiling::popRegion();


#elif defined(COMPADRE_USE_LAPACK)

    // requires column major input

    Kokkos::Profiling::pushRegion("QR::Setup(Create)");

    // find optimal blocksize when using LAPACK in order to allocate workspace needed
    int ipsec = 1, unused = -1, lwork = -1, info = 0; double wkopt = 0;

    dgels_( (char *)"N", &M, &N, &M, 
            (double *)NULL, &lda, 
            (double *)NULL, &ldb, 
            &wkopt, &lwork, &info );

    // size needed to malloc for each problem
    lwork = (int)wkopt;

    Kokkos::Profiling::popRegion();

    std::string transpose_or_no = "N";

    #ifdef LAPACK_DECLARED_THREADSAFE

        int scratch_space_size = scratch_vector_type::shmem_size( lwork );  // work space

        Kokkos::parallel_for(
            team_policy(num_matrices, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_space_size)),
            KOKKOS_LAMBDA (const member_type& teamMember) {

            scratch_vector_type scratch_work(teamMember.team_scratch(0), lwork);

            int i_info = 0;
        
            const int i = teamMember.league_rank();

            double * p_offset = P + i*lda*nda;
            double * rhs_offset = RHS + i*ldb*ndb;

            // use a custom # of neighbors for each problem, if possible
            const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
            int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i))*multiplier : M;

            Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
            dgels_( const_cast<char *>(transpose_or_no.c_str()), 
                    const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rows), 
                    p_offset, const_cast<int*>(&lda), 
                    rhs_offset, const_cast<int*>(&ldb), 
                    scratch_work.data(), const_cast<int*>(&lwork), &i_info);
            });

            if (i_info != 0)
              printf("\n dgels failed on index %d.", i);

        }, "QR Execution");

    #else

        double * scratch_work = (double *)malloc(sizeof(double)*lwork);
        for (int i=0; i<num_matrices; ++i) {

            int i_info = 0;
        
            double * p_offset = P + i*lda*nda;
            double * rhs_offset = RHS + i*ldb*ndb;

            // use a custom # of neighbors for each problem, if possible
            const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
            int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i))*multiplier : M;

            dgels_( const_cast<char *>(transpose_or_no.c_str()), 
                    const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rows), 
                    p_offset, const_cast<int*>(&lda), 
                    rhs_offset, const_cast<int*>(&ldb), 
                    scratch_work, const_cast<int*>(&lwork), &i_info);

            if (i_info != 0)
              printf("\n dgels failed on index %d.", i);
        }
        free(scratch_work);

    #endif // LAPACK is not threadsafe

#endif

}

void batchSVDFactorize(double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, const int num_matrices, const size_t max_neighbors, int * neighbor_list_sizes) {

// _U, _S, and _V are stored globally so that they can be used to apply targets in applySVD
// they are not needed on the CPU, only with CUDA because there is no dgelsd equivalent
// which is why these arguments are not used on the CPU

#ifdef COMPADRE_USE_CUDA

    const int NUM_STREAMS = 64;

    Kokkos::Profiling::pushRegion("SVD::Setup(Allocate)");

      int ldu = M;
      int ldv = N;
      int min_mn = (M<N) ? M : N;
      int max_mn = (M>N) ? M : N;
      // local U, S, and V must be allocated
      Kokkos::View<double*> U("U", num_matrices*ldu*M);
      Kokkos::View<double*> V("V", num_matrices*ldv*N);
      Kokkos::View<double*> S("S", num_matrices*min_mn);
      Kokkos::View<int*> devInfo("device info", num_matrices);

    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("SVD::Setup(Handle)");
      cudaError_t cudaStat1 = cudaSuccess;
      cusolverDnHandle_t * cusolver_handles = (cusolverDnHandle_t *) malloc(NUM_STREAMS*sizeof(cusolverDnHandle_t));
      cusolverStatus_t cusolver_stat = CUSOLVER_STATUS_SUCCESS;
      cudaStream_t *streams = (cudaStream_t *) malloc(NUM_STREAMS*sizeof(cudaStream_t));
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

    if (max_mn <= 32) { // batch version only works on small matrices

        Kokkos::Profiling::pushRegion("SVD::Workspace");

          int lwork = 0;

          cusolver_stat = cusolverDnDgesvdjBatched_bufferSize(
              cusolver_handles[0], jobz,
              M, N,
              P, lda, // P
              S.ptr_on_device(), // S
              U.ptr_on_device(), ldu, // U
              V.ptr_on_device(), ldv, // V
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
              S.ptr_on_device(),
              U.ptr_on_device(), ldu,
              V.ptr_on_device(), ldv,
              work.ptr_on_device(), lwork,
              devInfo.ptr_on_device(), gesvdj_params, num_matrices
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
              S.ptr_on_device(), // S
              U.ptr_on_device(), ldu, // U
              V.ptr_on_device(), ldv, // V
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
          Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA(const int i) {
              const int my_stream = i%NUM_STREAMS;

              cusolverDnDgesvdj(
                  cusolver_handles[my_stream], jobz, econ,
                  M, N,
                  P + i*lda*nda, lda,
                  S.ptr_on_device() + i*min_mn,
                  U.ptr_on_device() + i*ldu*M, ldu,
                  V.ptr_on_device() + i*ldv*N, ldv,
                  work.ptr_on_device() + i*lwork, lwork,
                  devInfo.ptr_on_device() + i, gesvdj_params
              );
  
//              cusolverDnDgesvd (
//                  cusolver_handles[my_stream],
//                  jobu,
//                  jobv,
//                  M,
//                  N,
//                  P + i*lda*nda,
//                  lda,
//                  S.ptr_on_device() + i*min_mn,
//                  U.ptr_on_device() + i*ldu*M,
//                  ldu,
//                  V.ptr_on_device() + i*ldv*N,
//                  ldv,
//                  work.ptr_on_device() + i*lwork,
//                  lwork,
//                  rwork.ptr_on_device() + i*(min_mn-1),
//                  devInfo.ptr_on_device() + i );
  
          });
        Kokkos::Profiling::popRegion();
    }

    cudaDeviceSynchronize();
    for (int i=0; i<NUM_STREAMS; ++i) {
        cusolverDnDestroy(cusolver_handles[i]);
    }

    free(streams);
    free(cusolver_handles);



    Kokkos::Profiling::pushRegion("SVD::S Copy");
      // deep copy neighbor list sizes over to gpu
      Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_neighbor_list_sizes(neighbor_list_sizes, num_matrices);
      Kokkos::View<int*> d_neighbor_list_sizes("neighbor list sizes on device", num_matrices);
      Kokkos::deep_copy(d_neighbor_list_sizes, h_neighbor_list_sizes);

      int scratch_space_size = 0;
      scratch_space_size += scratch_vector_type::shmem_size( min_mn );  // s
    Kokkos::Profiling::popRegion();
    
    Kokkos::parallel_for(
        team_policy(num_matrices, Kokkos::AUTO)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_space_size)), // shared memory
        KOKKOS_LAMBDA (const member_type& teamMember) {

        const int target_index = teamMember.league_rank();

        // use a custom # of neighbors for each problem, if possible
        const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
        int my_num_rows = d_neighbor_list_sizes(target_index)*multiplier;
        //int my_num_rows = d_neighbor_list_sizes(target_index)*multiplier : M;

    scratch_vector_type s(teamMember.team_scratch(0), min_mn ); // shared memory
        Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > RHS_(RHS + target_index*ldb*ndb, ldb, ldb);
        Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > U_(U.data() + target_index*ldu*M, ldu, M);
        Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > V_(V.data() + target_index*ldv*N, ldv, N);
        Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > S_(S.data() + target_index*min_mn, min_mn);

        // threshold for dropping singular values
        double S0 = S_(0);
        double eps = 1e-14;
        double abs_threshold = eps*S0;        

        // t1 takes on the role of S inverse
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, min_mn), [=] (const int i) {
            s(i) = ( std::abs(S_(i)) > abs_threshold ) ? 1./S_(i) : 0;
        });
        teamMember.team_barrier();

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
    }, "SVD: USV*RHS Multiply");





#elif defined(COMPADRE_USE_LAPACK)

    // later improvement could be to send in an optional view with the neighbor list size for each target to reduce work

    Kokkos::Profiling::pushRegion("SVD::Setup(Create)");
    double rcond = 1e-14; // rcond*S(0) is cutoff for singular values in dgelsd

    const int smlsiz = 25;
    const int nlvl = std::max(0, int( std::log2( std::min(M,N) / (smlsiz+1) ) + 1));
    const int liwork = 3*std::min(M,N)*nlvl + 11*std::min(M,N);

    int iwork[liwork];
    double s[std::max(M,N)];

    int lwork = -1;
    double wkopt = 0;
    int rank = 0;
    int info = 0;

    dgelsd_( &M, &N, &M, 
             P, &lda, 
             RHS, &ldb, 
             s, &rcond, &rank,
             &wkopt, &lwork, iwork, &info);
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

                double * p_offset = P + i*lda*nda;
                double * rhs_offset = RHS + i*ldb*ndb;

                // use a custom # of neighbors for each problem, if possible
                const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
                int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i))*multiplier : M;

                Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
                dgelsd_( const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rows), 
                         p_offset, const_cast<int*>(&lda), 
                         rhs_offset, const_cast<int*>(&ldb), 
                         scratch_s.data(), const_cast<double*>(&rcond), &i_rank,
                         scratch_work.data(), const_cast<int*>(&lwork), scratch_iwork.data(), &i_info);
                });

                if (i_info != 0)
                  printf("\n dgelsd failed on index %d.", i);

        }, "SVD Execution");

    #else
    
        double * scratch_work = (double *)malloc(sizeof(double)*lwork);
        double * scratch_s = (double *)malloc(sizeof(double)*std::max(M,N));
        int * scratch_iwork = (int *)malloc(sizeof(int)*liwork);
        
        for (int i=0; i<num_matrices; ++i) {

                int i_rank = 0;
                int i_info = 0;
        
                double * p_offset = P + i*lda*nda;
                double * rhs_offset = RHS + i*ldb*ndb;

                // use a custom # of neighbors for each problem, if possible
                const int multiplier = (max_neighbors > 0) ? M/max_neighbors : 1; // assumes M is some positive integer scalaing of max_neighbors
                int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i))*multiplier : M;

                dgelsd_( const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rows), 
                         p_offset, const_cast<int*>(&lda), 
                         rhs_offset, const_cast<int*>(&ldb), 
                         scratch_s, const_cast<double*>(&rcond), &i_rank,
                         scratch_work, const_cast<int*>(&lwork), scratch_iwork, &i_info);

                if (i_info != 0)
                  printf("\n dgelsd failed on index %d.", i);

        }

        free(scratch_work);
        free(scratch_s);
        free(scratch_iwork);

    #endif // LAPACK is not threadsafe

#endif
}

}; // GMLS_LinearAlgebra
}; // Compadre
