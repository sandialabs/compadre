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
        \param max_neighbors        [in] - integer for maximum neighbor over all targets
        \param neighbor_list_sizes  [in] - pointer to all neighbor list sizes for each target
    */
    void batchQRFactorize(ParallelManager pm, double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, int NRHS, const int num_matrices, const size_t max_neighbors = 0, const int initial_index_of_batch = 0, int * neighbor_list_sizes = NULL);

    /*! \brief Solves a batch of SVD problems

         P contains num_matrices * lda * ndb data which is num_matrices different matrices, and
         RHS contains num_matrices * ldn * ndb data which is num_matrices different matrix right hand sides.

        \param pm                   [in] - manager class for team and thread parallelism
        \param swap_layout_P        [in] - boolean to check if layout of P will need to be swapped or not
        \param P                [in/out] - evaluation of sampling functional on polynomial basis (in), meaningless workspace output (out)
        \param lda                  [in] - row dimension of each matrix in P
        \param nda                  [in] - columns dimension of each matrix in P
        \param swap_layout_RHS      [in] - boolean to check if layout of RHS will need to be swapped or not
        \param RHS              [in/out] - basis to invert P against (in), polynomial coefficients (out)
        \param ldb                  [in] - row dimension of each matrix in RHS
        \param ndb                  [in] - column dimension of each matrix in RHS
        \param M                    [in] - number of rows containing data (maximum rows over everything in batch)
        \param N                    [in] - number of columns containing data
        \param num_matrices         [in] - number of target (GMLS problems to be solved)
        \param max_neighbors        [in] - integer for maximum neighbor over all targets
        \param neighbor_list_sizes  [in] - pointer to all neighbor list sizes for each target
    */
    void batchSVDFactorize(ParallelManager pm, bool swap_layout_P, double *P, int lda, int nda, bool swap_layout_RHS, double *RHS, int ldb, int ndb, int M, int N, int NRHS, const int num_matrices, const size_t max_neighbors = 0, const int initial_index_of_batch = 0, int * neighbor_list_sizes = NULL);

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
        \param max_neighbors        [in] - integer for maximum neighbor over all targets
        \param neighbor_list_sizes  [in] - pointer to all neighbor list sizes for each target
    */
    void batchLUFactorize(ParallelManager pm, double *P, int lda, int nda, double *RHS, int ldb, int ndb, int M, int N, int NRHS, const int num_matrices, const size_t max_neighbors = 0, const int initial_index_of_batch = 0, int * neighbor_list_sizes = NULL);

} // GMLS_LinearAlgebra
} // Compadre

#endif
