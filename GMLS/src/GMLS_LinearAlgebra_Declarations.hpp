#ifndef _GMLS_LINEAR_ALGEBRA_DECLARATIONS_HPP_
#define _GMLS_LINEAR_ALGEBRA_DECLARATIONS_HPP_

#define BOOST_UBLAS_NDEBUG 1
#define USE_CUSTOM_SVD

#include "GMLS_Config.h"
#include <type_traits>
#include <thread>



#ifdef COMPADRE_USE_CUDA
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <cublas_api.h>
  #include <cusolverDn.h>
  #include <assert.h>
#endif



#ifdef COMPADRE_USE_LAPACK
  extern "C" int dgels_( char* trans, int *m, int *n, int *k, double *a, 
  			int *lda, double *c, int *ldc, double *work, int *lwork, int *info);
  
  extern "C" void dgelsd_( int* m, int* n, int* nrhs, double* a, int* lda,
  			double* b, int* ldb, double* s, double* rcond, int* rank,
  			double* work, int* lwork, int* iwork, int* info );
#ifdef COMPADRE_USE_OPENBLAS
  void openblas_set_num_threads(int num_threads);
#endif
#endif // COMPADRE_USE_LAPACK



#ifdef COMPADRE_USE_KOKKOSCORE
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

// KOKKOS TYPEDEFS

typedef Kokkos::TeamPolicy<>               team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;

typedef Kokkos::DefaultExecutionSpace::array_layout layout_type;

// reorders indices for layout of device
#ifdef COMPADRE_USE_CUDA
#define ORDER_INDICES(i,j) j,i
#else
#define ORDER_INDICES(i,j) i,j
#endif

typedef Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_matrix_type;
typedef Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_vector_type;
typedef Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_local_index_type;

typedef std::conditional<std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value,
                         Kokkos::LayoutLeft, Kokkos::LayoutRight>::type reverse_layout_type;

typedef uintptr_t device_ptr_type;

namespace Compadre {

namespace GMLS_LinearAlgebra {

	KOKKOS_INLINE_FUNCTION
	void createM(const member_type& teamMember, scratch_matrix_type M_data, scratch_matrix_type weighted_P, const int columns, const int rows);

	KOKKOS_INLINE_FUNCTION
	void orthogonalizeVectorBasis(const member_type& teamMember, scratch_matrix_type V);

	KOKKOS_INLINE_FUNCTION
	void largestTwoEigenvectorsThreeByThreeSymmetric(const member_type& teamMember, scratch_matrix_type V, scratch_matrix_type PtP, const int dimensions);

	void batchQRFactorize(double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, const int num_matrices, const size_t max_neighbors = 0, int * neighbor_list_sizes = NULL);

	void batchSVDFactorize(double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, const int num_matrices, const size_t max_neighbors = 0, int * neighbor_list_sizes = NULL);

}; // GMLS_LinearAlgebra
}; // Compadre

#endif
#endif


