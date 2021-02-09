#include <gtest/gtest.h>
#include <Compadre_KokkosParser.hpp>
#include "unittests/test_XYZ.hpp"
#include "unittests/test_NeighborLists.hpp"
#ifdef COMPADRE_USE_MPI
#include <mpi.h>
#endif

// this provides main(),
// but all other tests come from ./unittests/*.cpp
int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc, argv);
    // initializes MPI (if available) with command line arguments given
    #ifdef COMPADRE_USE_MPI
    MPI_Init(&argc, &argv);
    #endif
    auto kp = KokkosParser(argc, argv, true);
    const int sig = RUN_ALL_TESTS();
    // finalize Kokkos and MPI (if available)
    kp.finalize();
    #ifdef COMPADRE_USE_MPI
    MPI_Finalize();
    #endif
    return sig; 
}
