#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h>
#include <cstdio>
#include <random>

#include <Compadre_Config.h>
#include <Compadre_GMLS.hpp>
#include <Compadre_Evaluator.hpp>
#include <Compadre_PointCloudSearch.hpp>
#include <Compadre_KokkosParser.hpp>

#include "GMLS_Tutorial.hpp"

#ifdef COMPADRE_USE_MPI
#include <mpi.h>
#endif

#include <Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>

using namespace Compadre;

//! [Parse Command Line Arguments]

// called from command line
int main (int argc, char* args[]) {

#ifdef COMPADRE_USE_MPI
// initialized MPI (if avaialble) with command line arguments given
MPI_Init(&argc, &args);
#endif

// initializes Kokkos with command line arguments given
auto kp = KokkosParser(argc, args, true);

// becomes false if there is unwanted index in the filtered flags
bool all_passed = true;

// code block to reduce scope for all Kokkos View allocations
// otherwise, Views may be deallocating when we call Kokkos finalize() later
{
    // set the number of flags
    int num_flags = 200; // default 200 flags
    if (argc >= 2) {
        int arg2toi = atoi(args[1]);
        if (arg2toi > 0) {
            num_flags = arg2toi;
        }
    }
    //! [Parse Command Line Arguments]

    //! [Setting Up Data]
    Kokkos::Timer timer;
    Kokkos::Profiling::pushRegion("Setup Data");

    // create a view of flags
    Kokkos::View<int*, Kokkos::DefaultExecutionSpace> flags_device("flags", num_flags);
    Kokkos::View<int*>::HostMirror flags = Kokkos::create_mirror_view(flags_device);

    //! [Setting Up Data]

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Filter Data");

    //! [Filtering Data]

    // create arbitrary data
    int num_filtered_flags = 0; // number of filtered flags
    for (int i=0; i<num_flags; i++) {
        if ((i % 2) == 0) {
            flags(i) = 1;
            num_filtered_flags++;
        } else {
            flags(i) = 0;
        }
    }
    // copy the flags from host to device
    Kokkos::deep_copy(flags_device, flags);

    // Then call out the function to create view
    auto filtered_flags = filterViewByID<Kokkos::HostSpace>(flags_device, 1);

    //! [Filtering Data]

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Check Filtered Data");

    //! [Checking Filtered Data]

    if (filtered_flags.extent(0) != num_filtered_flags) {
        all_passed = false;
        std::cout << "Failed - number of filtered flags not matched!" << std::endl;
    }
    for (int i=0; i<filtered_flags.extent(0); i++) {
        if (filtered_flags(i) % 2 != 0) {
            all_passed = false;
            std::cout << "Failed - incorrect filtered flags " << filtered_flags(i) << std::endl;
        }
    }

    //! [Checking Filtered Data]
    // stop timing comparison loop
    Kokkos::Profiling::popRegion();
    //! [Finalize Program]

} // end of code block to reduce scope, causing Kokkos View de-allocations
// otherwise, Views may be deallocating when we call Kokkos finalize() later

// finalize Kokkos and MPI (if available)
kp.finalize();
#ifdef COMPADRE_USE_MPI
MPI_Finalize();
#endif

// output to user that test passed or failed
if (all_passed) {
    fprintf(stdout, "Passed test \n");
    return 0;
} else {
    fprintf(stdout, "Failed test \n");
    return -1;
}

} // main

//! [Finalize Program]
