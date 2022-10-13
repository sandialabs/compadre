#include <gtest/gtest.h>
#include <Compadre_KokkosParser.hpp>
#include "unittests/test_XYZ.hpp"
#include "unittests/test_NeighborLists.hpp"
#include "unittests/test_PointCloudSearch.hpp"
#include "unittests/test_LinearAlgebra.hpp"
#include "unittests/test_Targets.hpp"
#ifdef COMPADRE_USE_MPI
#include <mpi.h>
#endif

#define ASSERT_NO_DEATH(statement) \
ASSERT_EXIT({{ statement } ::exit(EXIT_SUCCESS); }, ::testing::ExitedWithCode(0), "")

//
// KokkosParser tests go here because they modify the Kokkos backend
//
TEST (KokkosInitialize, NoArgsGiven) { 
    ASSERT_NO_DEATH({
            // default constructor is hidden for KokkosParser
            // but still visible from this test
            auto kp = Compadre::KokkosParser(false);
    });
}
TEST (KokkosInitialize, NoCommandLineArgsGiven) { 
// KOKKOS_VERSION % 100 is the patch level
// KOKKOS_VERSION / 100 % 100 is the minor version
// KOKKOS_VERSION / 10000 is the major version
#ifdef KOKKOS_VERSION
  #define KOKKOS_VERSION_MAJOR KOKKOS_VERSION / 10000
  #define KOKKOS_VERSION_MINOR KOKKOS_VERSION / 100 % 100 
  #if KOKKOS_VERSION_MAJOR  < 4
    #if KOKKOS_VERSION_MINOR >= 7
            std::vector<std::string> arguments = {"--kokkos-num-threads=4"};
    #elif KOKKOS_VERSION_MINOR < 7
            std::vector<std::string> arguments = {"--kokkos-threads=4"};
    #endif
  #endif
#else // older version
            std::vector<std::string> arguments = {"--kokkos-threads=4"};
#endif
    ASSERT_NO_DEATH({
            auto kp = KokkosParser(arguments);
    });
}


// this provides main(),
// but all other tests come from ./unittests/*.cpp
int main(int argc, char **argv) {

    // initializes MPI (if available) with command line arguments given
    #ifdef COMPADRE_USE_MPI
    MPI_Init(&argc, &argv);
    #endif

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "Kokkos*";
    int sig = RUN_ALL_TESTS();

    // initializes kokkos
    Kokkos::initialize(argc, argv);

    // execute all tests
    ::testing::GTEST_FLAG(filter) = "-Kokkos*";
    sig += RUN_ALL_TESTS();

    // finalize Kokkos and MPI (if available)
    Kokkos::finalize();

    // finialize MPI (if available)
    #ifdef COMPADRE_USE_MPI
    MPI_Finalize();
    #endif
    return sig; 
}
