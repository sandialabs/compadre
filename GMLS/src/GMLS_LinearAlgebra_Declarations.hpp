#ifndef _GMLS_LINEAR_ALGEBRA_DECLARATIONS_HPP_
#define _GMLS_LINEAR_ALGEBRA_DECLARATIONS_HPP_

#define BOOST_UBLAS_NDEBUG 1
#define USE_CUSTOM_SVD

#include "GMLS_Config.h"
#include <type_traits>

#ifdef COMPADRE_USE_BOOST
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
typedef boost::numeric::ublas::matrix<double> boost_matrix_type;
typedef boost::numeric::ublas::matrix<double,boost::numeric::ublas::column_major> boost_matrix_fortran_type;
#endif

#ifdef COMPADRE_USE_LAPACK
extern "C" void dgesvd_( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info );
#endif

#ifdef COMPADRE_USE_KOKKOSCORE
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

// KOKKOS TYPEDEFS

typedef Kokkos::TeamPolicy<>               team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;

typedef Kokkos::DefaultExecutionSpace::array_layout layout_type;

typedef Kokkos::View<double**, layout_type, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_matrix_type;
typedef Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_vector_type;
typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_local_index_type;

namespace GMLS_LinearAlgebra {

	KOKKOS_INLINE_FUNCTION
	void GivensRotation(double& c, double& s, double a, double b);

	KOKKOS_INLINE_FUNCTION
	void createM(const member_type& teamMember, scratch_matrix_type M_data, scratch_matrix_type weighted_P, scratch_vector_type w, const int columns, const int rows);

	KOKKOS_INLINE_FUNCTION
	void computeSVD(const member_type& teamMember, scratch_matrix_type U, scratch_vector_type S, scratch_matrix_type Vt, scratch_matrix_type P, const int columns, const int rows);

    KOKKOS_INLINE_FUNCTION
    void invertM(const member_type& teamMember, scratch_vector_type y, scratch_matrix_type M_inv, scratch_matrix_type L, scratch_matrix_type M_data, const int columns);

	KOKKOS_INLINE_FUNCTION
	void upperTriangularBackSolve(const member_type& teamMember, scratch_matrix_type R, scratch_matrix_type Q, scratch_vector_type w, int columns, int rows);

	KOKKOS_INLINE_FUNCTION
	void GivensQR(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, int columns, int rows);

	KOKKOS_INLINE_FUNCTION
	void HouseholderQR(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, const int columns, const int rows);

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
    void GolubReinschSVD(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type U, scratch_matrix_type B, scratch_matrix_type V, const int columns, const int rows);

}

#endif
#endif


