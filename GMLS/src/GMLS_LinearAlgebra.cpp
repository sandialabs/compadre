#include "GMLS_LinearAlgebra_Definitions.hpp"

namespace GMLS_LinearAlgebra {

void batchQRFactorize(double *P, double *RHS, const size_t max_dim, const size_t num_rows, const size_t num_cols, const int num_matrices, const size_t max_neighbors, int * neighbor_list_sizes) {

    // quick calculations to determine # of rows needed for each problem
    int neighbor_num_multiplier;
    if (neighbor_list_sizes) {
        neighbor_num_multiplier = num_rows / max_neighbors;
    }

    int i_max_dim = static_cast<int>(max_dim);
    int i_num_rows = static_cast<int>(num_rows);
    int i_num_cols = static_cast<int>(num_cols);

    size_t dim_0 = max_dim;
    size_t dim_1 = num_cols;

#ifdef COMPADRE_USE_CUDA

    Kokkos::Profiling::pushRegion("QR::Setup(Pointers)");
    Kokkos::View<size_t*> array_P_RHS("P and RHS matrix pointers on device", 2*num_matrices);
    // get pointers to device data
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA(const int i) {
        array_P_RHS(i               ) = reinterpret_cast<size_t>(P   + i*dim_0*dim_0);
        array_P_RHS(i + num_matrices) = reinterpret_cast<size_t>(RHS + i*dim_0*dim_0);
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
    // call batched QR
    int info;
    //cublas_stat=cublasDgeqrfBatched(cublas_handle,
    //                         static_cast<int>(dim_0), static_cast<int>(dim_1), reinterpret_cast<double**>(array_P_Q_tau.ptr_on_device()),
    //                         static_cast<int>(dim_0),
    //                         reinterpret_cast<double**>(array_P_Q_tau.ptr_on_device()+num_matrices),
    //                         &info, num_matrices);
    cublas_stat=cublasDgelsBatched(cublas_handle,
                                   CUBLAS_OP_N, 
                                   static_cast<int>(dim_0),  // m
                                   static_cast<int>(dim_1),  // n
                                   static_cast<int>(dim_0),  // nrhs
                                   reinterpret_cast<double**>(array_P_RHS.ptr_on_device()),
                                   static_cast<int>(dim_0), // lda
                                   reinterpret_cast<double**>(array_P_RHS.ptr_on_device() + num_matrices),
                                   static_cast<int>(dim_0), // ldc
                                   &info, 
                                   devInfo,
                                   num_matrices );

    if (cublas_stat != CUBLAS_STATUS_SUCCESS)
      printf("\n cublasDgeqrfBatched failed");
    cudaDeviceSynchronize();
    Kokkos::Profiling::popRegion();


#elif defined(COMPADRE_USE_LAPACK)

    // requires column major input

    Kokkos::Profiling::pushRegion("QR::Setup(Create)");

    // find optimal blocksize when using LAPACK in order to allocate workspace needed
    int ipsec = 1, unused = -1, lwork = -1, info = 0; double wkopt = 0;

    dgels_( (char *)"N", &i_num_rows, &i_num_cols, &i_num_rows, 
            (double *)NULL, &i_max_dim, 
            (double *)NULL, &i_num_rows, 
            &wkopt, &lwork, &info );

    // size needed to malloc for each problem
    lwork = (int)wkopt;

    printf("%d is opt: lapack_opt_blocksize\n", lwork);
    //Kokkos::View<double*> work("work space for lapack", num_matrices*lwork);

    //Kokkos::View<int*> infos("info", num_matrices);
    //Kokkos::deep_copy(infos, 0);
    Kokkos::Profiling::popRegion();

    //double *work = (double *)malloc(num_matrices*lwork*sizeof(double));
    // get OMP_NUM_THREADS to store as a reference, then set to 1
//#if defined(_OPENMP)
//	unsigned int thread_qty;
//    try {
//        std::string omp_string = std::getenv("OMP_NUM_THREADS");
//    	thread_qty = std::max(atoi(omp_string.c_str()), 1);
//	} catch (...) {
//    	thread_qty = 1;
//	}
//    omp_set_num_threads(1);
//    printf("%d threads called.\n", thread_qty);
//#endif
//
//#ifdef Compadre_USE_OPENBLAS
//   int openblas_num_threads = 1;
//   set_openblas_num_threads(openblas_num_threads);
//   set_openmp_num_threads(openblas_num_threads);
//   printf("Reduce openblas threads to 1.\n");
//#endif

    int scratch_space_size = scratch_vector_type::shmem_size( lwork );  // work space
    std::string transpose_or_no = "N";

    Kokkos::parallel_for(
        team_policy(num_matrices, Kokkos::AUTO)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_space_size)),
        KOKKOS_LAMBDA (const member_type& teamMember) {

            scratch_vector_type scratch_work(teamMember.team_scratch(0), lwork);

            int i_info = 0;
    
            const int i = teamMember.league_rank();

            double * p_offset = P + i*max_dim*max_dim;
            double * rhs_offset = RHS + i*num_rows*num_rows;

            // use a custom # of neighbors for each problem, if possible
            int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i))*neighbor_num_multiplier : i_num_rows;

            dgels_( const_cast<char *>(transpose_or_no.c_str()), &my_num_rows, const_cast<int*>(&i_num_cols), &my_num_rows, 
                    p_offset, const_cast<int*>(&i_max_dim), 
                    rhs_offset, const_cast<int*>(&i_num_rows), 
                    scratch_work.data(), const_cast<int*>(&lwork), &i_info);

            //dgelsd_( &my_num_rows, const_cast<int*>(&i_num_cols), &my_num_rows, 
            //         p_offset, const_cast<int*>(&i_max_dim), 
            //         rhs_offset, const_cast<int*>(&i_num_rows), 
            //         scratch_s.data(), const_cast<double*>(&rcond), &i_rank,
            //         scratch_work.data(), const_cast<int*>(&lwork), scratch_iwork.data(), &i_info);
            //dgelsd_( const_cast<int*>(&i_num_rows), const_cast<int*>(&i_num_cols), const_cast<int*>(&i_num_rows), 
            //         p_offset, const_cast<int*>(&i_max_dim), 
            //         rhs_offset, const_cast<int*>(&i_num_rows), 
            //         scratch_s.data(), const_cast<double*>(&rcond), &i_rank,
            //         scratch_work.data(), const_cast<int*>(&lwork), scratch_iwork.data(), &i_info);

    }, "QR Execution");

    //Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA (const int i) {
    //        int t_dim_0 = dim_0;
    //        int t_dim_1 = dim_1;
    //        int t_info = infos[i];
    //        int t_lwork = lwork;
    //        double * p_offset = P + i*dim_0*dim_0;
    //        double * rhs_offset = RHS + i*dim_0*dim_0; double * work_offset = work.data() + i*lwork;
    //        dgels_( const_cast<char *>(transpose_or_no.c_str()), &t_dim_0, &t_dim_1, &t_dim_0, 
    //                p_offset, &t_dim_0, 
    //                rhs_offset, &t_dim_0, 
    //                work_offset, &t_lwork, &t_info);
    //});

#endif

}

void batchSVDFactorize(double *P, double *RHS, const size_t max_dim, const size_t num_rows, const size_t num_cols, const int num_matrices, const size_t max_neighbors, int * neighbor_list_sizes) {
#if defined(COMPADRE_USE_LAPACK)


    // quick calculations to determine # of rows needed for each problem
    int neighbor_num_multiplier;
    if (neighbor_list_sizes) {
        neighbor_num_multiplier = num_rows / max_neighbors;
    }

    // later improvement could be to send in an optional view with the neighbor list size for each target to reduce work

    Kokkos::Profiling::pushRegion("SVD::Setup(Create)");
    double rcond = 1e-14; // rcond*S(0) is cutoff for singular values in dgelsd

    const int smlsiz = 25;
    const int nlvl = std::max(0, int( std::log2( std::min(num_rows,num_cols) / (smlsiz+1) ) + 1));
    const int liwork = 3*std::min(num_rows,num_cols)*nlvl + 11*std::min(num_rows,num_cols);

    int iwork[liwork];
    double s[max_dim];

    int lwork = -1;
    double wkopt = 0;
    int rank = 0;
    int info = 0;

    int i_max_dim = static_cast<int>(max_dim);
    int i_num_rows = static_cast<int>(num_rows);
    int i_num_cols = static_cast<int>(num_cols);

    dgelsd_( &i_num_rows, &i_num_cols, &i_num_rows, 
             P, &i_max_dim, 
             RHS, &i_num_rows, 
             s, &rcond, &rank,
             &wkopt, &lwork, iwork, &info);
    lwork = (int)wkopt;


    //Kokkos::View<int*> all_iwork("iwork for all calls", num_matrices*liwork);
    //Kokkos::View<double*> all_work("work for all calls", num_matrices*lwork);
    //Kokkos::View<double*> all_s("s for all calls", num_matrices*max_dim);
    //Kokkos::parallel_for(
    //    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_matrices),
    //    KOKKOS_LAMBDA (const int i) {

    //            int i_rank = 0;
    //            int i_info = 0;
    //
    //            double * p_offset = P + i*max_dim*max_dim;
    //            double * rhs_offset = RHS + i*num_rows*num_rows;

    //            int * scratch_iwork = all_iwork.data() + i*liwork;
    //            double * scratch_work = all_work.data() + i*lwork;
    //            double * scratch_s = all_s.data() + i*max_dim;

    //            dgelsd_( const_cast<int*>(&i_num_rows), const_cast<int*>(&i_num_cols), const_cast<int*>(&i_num_rows), 
    //                     p_offset, const_cast<int*>(&i_max_dim), 
    //                     rhs_offset, const_cast<int*>(&i_num_rows), 
    //                     scratch_s, const_cast<double*>(&rcond), &i_rank,
    //                     scratch_work, const_cast<int*>(&lwork), scratch_iwork, &i_info);
    //}, "SVD Execution");

    // works
    int scratch_space_size = 0;
    scratch_space_size += scratch_vector_type::shmem_size( lwork );  // work space
    scratch_space_size += scratch_vector_type::shmem_size( max_dim );  // s
    scratch_space_size += scratch_local_index_type::shmem_size( liwork ); // iwork space

    Kokkos::Profiling::popRegion();
    
    
    Kokkos::parallel_for(
        team_policy(num_matrices, Kokkos::AUTO)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_space_size)),
        KOKKOS_LAMBDA (const member_type& teamMember) {

            scratch_vector_type scratch_work(teamMember.team_scratch(0), lwork);
            scratch_vector_type scratch_s(teamMember.team_scratch(0), max_dim);
            scratch_local_index_type scratch_iwork(teamMember.team_scratch(0), liwork);

            int i_rank = 0;
            int i_info = 0;
    
            const int i = teamMember.league_rank();

            double * p_offset = P + i*max_dim*max_dim;
            double * rhs_offset = RHS + i*num_rows*num_rows;

            // use a custom # of neighbors for each problem, if possible
            int my_num_rows = (neighbor_list_sizes) ? (*(neighbor_list_sizes + i))*neighbor_num_multiplier : i_num_rows;

            dgelsd_( &my_num_rows, const_cast<int*>(&i_num_cols), &my_num_rows, 
                     p_offset, const_cast<int*>(&i_max_dim), 
                     rhs_offset, const_cast<int*>(&i_num_rows), 
                     scratch_s.data(), const_cast<double*>(&rcond), &i_rank,
                     scratch_work.data(), const_cast<int*>(&lwork), scratch_iwork.data(), &i_info);
            //dgelsd_( const_cast<int*>(&i_num_rows), const_cast<int*>(&i_num_cols), const_cast<int*>(&i_num_rows), 
            //         p_offset, const_cast<int*>(&i_max_dim), 
            //         rhs_offset, const_cast<int*>(&i_num_rows), 
            //         scratch_s.data(), const_cast<double*>(&rcond), &i_rank,
            //         scratch_work.data(), const_cast<int*>(&lwork), scratch_iwork.data(), &i_info);

    }, "SVD Execution");

#endif
}

//void batchLUFactorize(double *P, double *RHS, const size_t dim_0, const size_t dim_1, const int num_matrices) {
//}

};
