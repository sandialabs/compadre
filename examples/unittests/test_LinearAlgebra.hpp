#ifndef TEST_LINEARALGEBRA
#define TEST_LINEARALGEBRA

#include "Compadre_LinearAlgebra_Definitions.hpp"
#include "Compadre_LinearAlgebra_Declarations.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace Compadre;

class LinearAlgebraTest: public ::testing::Test {
public:

    ParallelManager pm;
    Kokkos::View<double*, host_execution_space> A;
    Kokkos::View<double*, host_execution_space> B;

    LinearAlgebraTest( ) {
        // initialization
        pm = ParallelManager();
    }

    void SetUp(const int lda, const int nda, const int ldb, const int ndb, 
            const int M, const int N, const int NRHS, const int num_matrices, 
            const int rank) {

        // multiple full rank matrices
        Kokkos::resize(A, lda*nda*num_matrices);
        Kokkos::deep_copy(A, 0.0);
        Kokkos::resize(B, ldb*ndb*num_matrices);
        Kokkos::deep_copy(B, 0.0);

        if (M==N) { // square case
            if (rank==M) { // full rank
                // fill in A matrices
                std::vector<int>    a_i ={0,0,1,1,1,2,2};
                std::vector<int>    a_j ={0,1,0,1,2,1,2};
                std::vector<double> a_k ={-4,1,1,-4,1,1,-4};

                for (int mat_num=0; mat_num<num_matrices; ++mat_num) {
                    host_scratch_matrix_right_type this_A(A.data() + mat_num*lda*nda, lda, nda);
                    for (int ind=0; ind<a_i.size(); ++ind) {
                        this_A(a_i[ind], a_j[ind]) = a_k[ind];
                    }
                }

                // B 
                std::vector<int> b_i, b_j;
                std::vector<double> b_k;
                if (NRHS==M) {
                    b_i ={0,1,2};
                    b_j ={0,1,2};
                    b_k ={0,1,2};
                } else {
                    b_i ={0,1,2,2};
                    b_j ={0,1,2,3};
                    b_k ={0,1,2,3};
                }
                for (int mat_num=0; mat_num<num_matrices; ++mat_num) {
                    // note that ndb and ldb ordering is reversed
                    // this is because ldb and ndb are specified w.r.t. a layout right matrix
                    // which when viewed as layout left swaps the ordering
                    host_scratch_matrix_left_type this_B(B.data() + mat_num*ldb*ndb, ndb, ldb);
                    for (int ind=0; ind<b_i.size(); ++ind) {
                        this_B(b_i[ind], b_j[ind]) = b_k[ind];
                    }
                }
            }
        }
    }

    void TearDown( ) {
        // after test completes
    }
};

TEST_F (LinearAlgebraTest, Square_FullRank_batchLUFactorize_Same_LDA_NDA) {
    int M=3, N=3, NRHS=3, num_matrices=2, rank=3;
    int lda=3, nda=3;
    int ldb=3, ndb=3; // relative to layout right
    SetUp(lda, nda, ldb, ndb, M, N, NRHS, num_matrices, rank);
    // LU expects layout left B, so ndb and ldb reverse ordered
    GMLS_LinearAlgebra::batchLUFactorize(pm, A.data(), lda, nda, B.data(), ndb, ldb, M, N, NRHS, num_matrices);
    // solution: X = [
    //                   0  -0.071428571428571  -0.035714285714286
    //                   0  -0.285714285714286  -0.142857142857143
    //                   0  -0.071428571428571  -0.535714285714286
    //
    //               0  -0.071428571428571  -0.035714285714286  -0.053571428571429
    //               0  -0.285714285714286  -0.142857142857143  -0.214285714285714
    //               0  -0.071428571428571  -0.535714285714286  -0.803571428571429
    //               ]
    //for (int i=0; i<3; ++i) {
    //    for (int j=0; j<3; ++j) {
    //        printf("B(%d,%d): %.16f\n", i, j, B(i*N+j));
    //    }
    //}
    EXPECT_NEAR(               0.0, B(0*3+0), 1e-14);
    EXPECT_NEAR(               0.0, B(1*3+0), 1e-14);
    EXPECT_NEAR(               0.0, B(2*3+0), 1e-14);
    EXPECT_NEAR(-0.071428571428571, B(0*3+1), 1e-14);
    EXPECT_NEAR(-0.285714285714286, B(1*3+1), 1e-14);
    EXPECT_NEAR(-0.071428571428571, B(2*3+1), 1e-14);
    EXPECT_NEAR(-0.035714285714286, B(0*3+2), 1e-14);
    EXPECT_NEAR(-0.142857142857143, B(1*3+2), 1e-14);
    EXPECT_NEAR(-0.535714285714286, B(2*3+2), 1e-14);
}

TEST_F (LinearAlgebraTest, Square_FullRank_batchLUFactorize_Larger_LDA_NDA) {
    // lda and nda larger than M and N
    int M=3, N=3, NRHS=3, num_matrices=2, rank=3;
    int lda=7, nda=12;
    int ldb=3, ndb=3; // relative to layout right
    SetUp(lda, nda, ldb, ndb, M, N, NRHS, num_matrices, rank);
    // LU expects layout left B, so ndb and ldb reverse ordered
    GMLS_LinearAlgebra::batchLUFactorize(pm, A.data(), lda, nda, B.data(), ndb, ldb, M, N, NRHS, num_matrices);
    EXPECT_NEAR(               0.0, B(0*3+0), 1e-14);
    EXPECT_NEAR(               0.0, B(1*3+0), 1e-14);
    EXPECT_NEAR(               0.0, B(2*3+0), 1e-14);
    EXPECT_NEAR(-0.071428571428571, B(0*3+1), 1e-14);
    EXPECT_NEAR(-0.285714285714286, B(1*3+1), 1e-14);
    EXPECT_NEAR(-0.071428571428571, B(2*3+1), 1e-14);
    EXPECT_NEAR(-0.035714285714286, B(0*3+2), 1e-14);
    EXPECT_NEAR(-0.142857142857143, B(1*3+2), 1e-14);
    EXPECT_NEAR(-0.535714285714286, B(2*3+2), 1e-14);
}

TEST_F (LinearAlgebraTest, Square_FullRank_batchLUFactorize_Larger_LDB_NDB) {
    // lda and nda larger than M and N
    int M=3, N=3, NRHS=3, num_matrices=2, rank=3;
    int lda=3, nda=3;
    int ldb=7, ndb=13; // relative to layout right
    SetUp(lda, nda, ldb, ndb, M, N, NRHS, num_matrices, rank);
    // LU expects layout left B, so ndb and ldb reverse ordered
    GMLS_LinearAlgebra::batchLUFactorize(pm, A.data(), lda, nda, B.data(), ndb, ldb, M, N, NRHS, num_matrices);
    EXPECT_NEAR(               0.0, B(0*ldb+0), 1e-14);
    EXPECT_NEAR(               0.0, B(1*ldb+0), 1e-14);
    EXPECT_NEAR(               0.0, B(2*ldb+0), 1e-14);
    EXPECT_NEAR(-0.071428571428571, B(0*ldb+1), 1e-14);
    EXPECT_NEAR(-0.285714285714286, B(1*ldb+1), 1e-14);
    EXPECT_NEAR(-0.071428571428571, B(2*ldb+1), 1e-14);
    EXPECT_NEAR(-0.035714285714286, B(0*ldb+2), 1e-14);
    EXPECT_NEAR(-0.142857142857143, B(1*ldb+2), 1e-14);
    EXPECT_NEAR(-0.535714285714286, B(2*ldb+2), 1e-14);
}
//TEST_F (LinearAlgebraTest, SquareRankDeficient) {
//}
//TEST_F (LinearAlgebraTest, RectangularFullRank) {
//}
//TEST_F (LinearAlgebraTest, RectangularRankDeficient) {
//}

#endif
