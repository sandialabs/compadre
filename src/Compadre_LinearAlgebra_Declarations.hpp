#ifndef _COMPADRE_LINEAR_ALGEBRA_DECLARATIONS_HPP_
#define _COMPADRE_LINEAR_ALGEBRA_DECLARATIONS_HPP_

#include "Compadre_Config.h"
#include "Compadre_Typedefs.hpp"
#include "Compadre_ParallelManager.hpp"

namespace Compadre {

namespace GMLS_LinearAlgebra {

    /*! \brief Calculates two eigenvectors corresponding to two dominant eigenvalues
        \param teamMember    [in] - Kokkos::TeamPolicy member type (created by parallel_for)
        \param V            [out] - dimension * dimension Kokkos View
        \param PtP           [in] - dimension * dimension Kokkos View
        \param dimensions    [in] - dimension of PtP
    */
    KOKKOS_INLINE_FUNCTION
    void largestTwoEigenvectorsThreeByThreeSymmetric(const member_type& teamMember, scratch_matrix_right_type V, scratch_matrix_right_type PtP, const int dimensions, pool_type& random_number_pool);

    /*! \brief Solves a batch of QR problems

         P contains num_matrices * lda * ndb data which is num_matrices different matrices, and
         RHS contains num_matrices * ldn * ndb data which is num_matrices different matrix right hand sides.

        \param pm                   [in] - manager class for team and thread parallelism
        \param P                [in/out] - evaluation of sampling functional on polynomial basis (in), meaningless workspace output (out)
        \param lda                  [in] - row dimension of each matrix in P
        \param nda                  [in] - columns dimension of each matrix in P
        \param RHS              [in/out] - basis to invert P against (in), polynomial coefficients (out)
        \param ldb                  [in] - row dimension of each matrix in RHS
        \param ndb                  [in] - column dimension of each matrix in RHS
        \param M                    [in] - number of rows containing data (maximum rows over everything in batch)
        \param N                    [in] - number of columns containing data
        \param num_matrices         [in] - number of target (GMLS problems to be solved)
    */
    void batchQRFactorize(ParallelManager pm, double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, int NRHS, const int num_matrices);

    /*! \brief Solves a batch of SVD problems

         P contains num_matrices * lda * ndb data which is num_matrices different matrices, and
         RHS contains num_matrices * ldn * ndb data which is num_matrices different matrix right hand sides.

        \param pm                   [in] - manager class for team and thread parallelism
        \param P                [in/out] - evaluation of sampling functional on polynomial basis (in), meaningless workspace output (out)
        \param lda                  [in] - row dimension of each matrix in P
        \param nda                  [in] - columns dimension of each matrix in P
        \param RHS              [in/out] - basis to invert P against (in), polynomial coefficients (out)
        \param ldb                  [in] - row dimension of each matrix in RHS
        \param ndb                  [in] - column dimension of each matrix in RHS
        \param M                    [in] - number of rows containing data (maximum rows over everything in batch)
        \param N                    [in] - number of columns containing data
        \param num_matrices         [in] - number of target (GMLS problems to be solved)
    */
    void batchSVDFactorize(ParallelManager pm, double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, int NRHS, const int num_matrices);

    /*! \brief Solves a batch of LU problems

         P contains num_matrices * lda * ndb data which is num_matrices different matrices, and
         RHS contains num_matrices * ldn * ndb data which is num_matrices different matrix right hand sides.

        \param pm                   [in] - manager class for team and thread parallelism
        \param P                [in/out] - evaluation of sampling functional on polynomial basis (in), meaningless workspace output (out)
        \param lda                  [in] - row dimension of each matrix in P
        \param nda                  [in] - columns dimension of each matrix in P
        \param RHS              [in/out] - basis to invert P against (in), polynomial coefficients (out)
        \param ldb                  [in] - row dimension of each matrix in RHS
        \param ndb                  [in] - column dimension of each matrix in RHS
        \param M                    [in] - number of rows containing data (maximum rows over everything in batch)
        \param N                    [in] - number of columns containing data
        \param num_matrices         [in] - number of target (GMLS problems to be solved)
    */
    void batchLUFactorize(ParallelManager pm, double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, int NRHS, const int num_matrices);

} // GMLS_LinearAlgebra
} // Compadre

#endif
