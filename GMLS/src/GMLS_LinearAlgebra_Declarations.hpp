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

#ifdef COMPADRE_USE_BOOST
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
typedef boost::numeric::ublas::matrix<double> boost_matrix_type;
typedef boost::numeric::ublas::matrix<double,boost::numeric::ublas::column_major> boost_matrix_fortran_type;
#endif

#ifdef COMPADRE_USE_LAPACK

// forms U,s,V from A
extern "C" void dgesvd_( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info );

// forms QR from A
extern "C" int dgeqrf_(int *m, int *n, double *a, int *lda,
	        double *tau, double *work, int *lwork, int *info);

// forms Q from QR reflectors
extern "C" int dormqr_( char* side, char* trans, int *m, int *n, int *k, double *a, 
                int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);

extern "C" int dgels_( char* trans, int *m, int *n, int *k, double *a, 
                int *lda, double *c, int *ldc, double *work, int *lwork, int *info);

//extern "C" int dgelsd_( int *m, int *n, int *k, double *a, int *lda, double *c, int *ldc, 
//                double *s, double *rcond, int *rank, double *work, int *lwork, int *iwork, int *info);
extern "C" void dgelsd_( int* m, int* n, int* nrhs, double* a, int* lda,
                double* b, int* ldb, double* s, double* rcond, int* rank,
                double* work, int* lwork, int* iwork, int* info );

#ifdef COMPADRE_USE_OPENBLAS
//#include <cblas.h>
//extern "C" void openblas_set_num_threads(int);
void openblas_set_num_threads(int num_threads);
#endif
#endif

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

//typedef Kokkos::View<double**, layout_type, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_matrix_type;
typedef Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_matrix_type;
typedef Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_matrix_type_layout_left;
//typedef Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > device_matrix_type;
//typedef Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_vector_type;
typedef Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_vector_type;
//typedef Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > device_vector_type;
//typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_local_index_type;
typedef Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_local_index_type;
//typedef Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > device_local_index_type;

typedef std::conditional<std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value,
                         Kokkos::LayoutLeft, Kokkos::LayoutRight>::type reverse_layout_type;

//typedef std::conditional<std::is_same<std::intptr_t, std::int32_t>::value,
//                         float, double>::type device_ptr_type;
typedef uintptr_t device_ptr_type;

namespace GMLS_LinearAlgebra {

	KOKKOS_INLINE_FUNCTION
	void GivensRotation(double& c, double& s, double a, double b);

	KOKKOS_INLINE_FUNCTION
	void createM(const member_type& teamMember, scratch_matrix_type M_data, scratch_matrix_type weighted_P, const int columns, const int rows);

	KOKKOS_INLINE_FUNCTION
	void computeSVD(const member_type& teamMember, scratch_matrix_type U, scratch_vector_type S, scratch_matrix_type Vt, scratch_matrix_type P, int columns, int rows);

    KOKKOS_INLINE_FUNCTION
    void invertM(const member_type& teamMember, scratch_vector_type y, scratch_matrix_type M_inv, scratch_matrix_type L, scratch_matrix_type M_data, const int columns);

	KOKKOS_INLINE_FUNCTION
	void upperTriangularBackSolve(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, scratch_vector_type w, int columns, int rows);

	KOKKOS_INLINE_FUNCTION
	void constructQ(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, scratch_vector_type tau, int columns, int rows);

	KOKKOS_INLINE_FUNCTION
	void GivensQR(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_vector_type t3, scratch_matrix_type Q, scratch_matrix_type R, int columns, int rows);

	KOKKOS_INLINE_FUNCTION
	void HouseholderQR(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, scratch_vector_type tau, const int columns, const int rows);

    KOKKOS_INLINE_FUNCTION
    double closestEigenvalueTwoByTwoEigSolver(const member_type& teamMember, double comparison_value, double a11, double a12, double a21, double a22);

    KOKKOS_INLINE_FUNCTION
    void orthogonalizeVectorBasis(const member_type& teamMember, scratch_matrix_type V);

	KOKKOS_INLINE_FUNCTION
	void largestTwoEigenvectorsThreeByThreeSymmetric(const member_type& teamMember, scratch_matrix_type V, scratch_matrix_type PtP, const int dimensions);

	KOKKOS_INLINE_FUNCTION
	void GivensBidiagonalReduction(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type U, scratch_matrix_type B, scratch_matrix_type V, const int columns, const int rows);

	KOKKOS_INLINE_FUNCTION
	void HouseholderBidiagonalReduction(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type U, scratch_matrix_type B, scratch_matrix_type V, const int columns, const int rows);

    KOKKOS_INLINE_FUNCTION
    void GolubKahanSVD(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type U, scratch_matrix_type B, scratch_matrix_type V, const int B2b, const int B2e, const int columns, const int rows);

    KOKKOS_INLINE_FUNCTION
    void GolubReinschSVD(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type B, scratch_matrix_type U, scratch_vector_type S, scratch_matrix_type V, const int columns, const int rows);

    KOKKOS_INLINE_FUNCTION
    void matrixToLayoutLeft(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type A, const int columns, const int rows);

    void batchQRFactorize(double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, const int num_matrices, const size_t max_neighbors = 0, int * neighbor_list_sizes = NULL);

    void batchSVDFactorize(double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, const int num_matrices, const size_t max_neighbors = 0, int * neighbor_list_sizes = NULL);

    //void batchLUFactorize(double *P, double *RHS, const size_t dim_0, const size_t dim_1, const int num_matrices);
}

#endif
#endif


