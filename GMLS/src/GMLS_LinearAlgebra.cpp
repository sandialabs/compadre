#include "GMLS_LinearAlgebra_Definitions.hpp"

namespace GMLS_LinearAlgebra {

void batchQRFactorize(double *P, double *RHS, const size_t dim_0, const size_t dim_1, const int num_matrices) {
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
    int ipsec = 1, unused = -1, bnp = static_cast<int>(dim_1), lwork = -1, info = 0; double wkopt = 0;
    int mmd = static_cast<int>(dim_0);
    dgels_( (char *)"N", &mmd, &bnp, &mmd, 
            (double *)NULL, &mmd, 
            (double *)NULL, &mmd, 
            &wkopt, &lwork, &info );

    // size needed to malloc for each problem
    lwork = (int)wkopt;
    printf("%d is opt: lapack_opt_blocksize\n", lwork);

    double *work = (double *)malloc(num_matrices*lwork*sizeof(double));
    Kokkos::View<int*> infos("info", num_matrices);
    Kokkos::deep_copy(infos, 0);
    Kokkos::Profiling::popRegion();

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

	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA (const int i) {
                int t_dim_0 = dim_0;
                int t_dim_1 = dim_1;
                int t_info = infos[i];
                int t_lwork = lwork;
                double * p_offset = P + i*dim_0*dim_0;
                double * rhs_offset = RHS + i*dim_0*dim_0;
                double * work_offset = work + i*lwork;
                dgels_( (char *)"N", &t_dim_0, &t_dim_1, &t_dim_0, 
                        p_offset, &t_dim_0, 
                        rhs_offset, &t_dim_0, 
                        work_offset, &t_lwork, &t_info);
	});


    Kokkos::Profiling::pushRegion("QR::Setup(Cleanup)");
    free(work);
    Kokkos::Profiling::popRegion();

#endif

}


//void batchLUFactorize(double *P, double *RHS, const size_t dim_0, const size_t dim_1, const int num_matrices) {
//}

};
