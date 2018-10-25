#include "GMLS_LinearAlgebra_Definitions.hpp"
namespace GMLS_LinearAlgebra {

void batchQRFactorize(double *P, double *RHS, int M, int N, int lda, int ldb, const int num_matrices, const size_t max_neighbors, int * neighbor_list_sizes) {

#ifdef COMPADRE_USE_CUDA

    Kokkos::Profiling::pushRegion("QR::Setup(Pointers)");
    Kokkos::View<size_t*> array_P_RHS("P and RHS matrix pointers on device", 2*num_matrices);

    // get pointers to device data
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA(const int i) {
        array_P_RHS(i               ) = reinterpret_cast<size_t>(P   + i*lda*lda);
        array_P_RHS(i + num_matrices) = reinterpret_cast<size_t>(RHS + i*ldb*ldb);
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
    int scratch_space_size = scratch_vector_type::shmem_size( lwork );  // work space

    Kokkos::parallel_for(
        team_policy(num_matrices, Kokkos::AUTO)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_space_size)),
        KOKKOS_LAMBDA (const member_type& teamMember) {

            scratch_vector_type scratch_work(teamMember.team_scratch(0), lwork);

            int i_info = 0;
    
            const int i = teamMember.league_rank();

            double * p_offset = P + i*lda*lda;
            double * rhs_offset = RHS + i*ldb*ldb;

            // use a custom # of neighbors for each problem, if possible
            const int multiplier = M/max_neighbors; // assumes M is some positive integer scalaing of max_neighbors
            int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i))*multiplier : M;

            dgels_( const_cast<char *>(transpose_or_no.c_str()), 
                    const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rows), 
                    p_offset, const_cast<int*>(&lda), 
                    rhs_offset, const_cast<int*>(&ldb), 
                    scratch_work.data(), const_cast<int*>(&lwork), &i_info);

            if (i_info != 0)
              printf("\n dgels failed on index %d.", i);

    }, "QR Execution");

#endif

}

void batchSVDFactorize(double *P, double *RHS, int M, int N, int lda, int ldb, const int num_matrices, const size_t max_neighbors, int * neighbor_list_sizes) {
#if defined(COMPADRE_USE_LAPACK)

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

            double * p_offset = P + i*lda*lda;
            double * rhs_offset = RHS + i*ldb*ldb;

            // use a custom # of neighbors for each problem, if possible
            const int multiplier = M/max_neighbors; // assumes M is some positive integer scalaing of max_neighbors
            int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i))*multiplier : M;

            dgelsd_( const_cast<int*>(&my_num_rows), const_cast<int*>(&N), const_cast<int*>(&my_num_rows), 
                     p_offset, const_cast<int*>(&lda), 
                     rhs_offset, const_cast<int*>(&ldb), 
                     scratch_s.data(), const_cast<double*>(&rcond), &i_rank,
                     scratch_work.data(), const_cast<int*>(&lwork), scratch_iwork.data(), &i_info);

            if (i_info != 0)
              printf("\n dgelsd failed on index %d.", i);

    }, "SVD Execution");

#endif
}

};
