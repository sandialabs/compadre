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
#include <Compadre_PointData.hpp>

#include "GMLS_Tutorial.hpp"

#ifdef COMPADRE_USE_MPI
#include <mpi.h>
#endif

#include <Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>

using namespace Compadre;

class MyFunctor {

    public:

    //const PointData<Kokkos::View<double**> >& t_bah;
    PointData<Kokkos::View<double**> >* t_bah;

    struct Test{};

    Kokkos::View<double*> vals;

    MyFunctor(PointData<Kokkos::View<double**> >* _t_bah) : t_bah(_t_bah) {
        vals = Kokkos::View<double*>("vals", 100);
        Kokkos::deep_copy(vals, 0.0);
        Kokkos::fence();
    }

    //KOKKOS_INLINE_FUNCTION
    //void operator() (const Test&, const int i) const;
    KOKKOS_INLINE_FUNCTION
    void operator() (const Test&, const int i) const {
        //auto m_t_bah = *(const_cast<PointData<Kokkos::View<double**> >* >(&t_bah));
        vals(i) += 10;
        t_bah->setValueOnHost(i,0,10);
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const int i) const {
        // unconst
        //auto m_t_bah = *(const_cast<PointData<Kokkos::View<double**> >* >(&t_bah));
        vals(i) += 1;
        t_bah->setValueOnHost(i,0,2);
    }

};

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
    // set the number of columns
    int num_cols = 50; // default 50 columns
    if (argc >= 3) {
        int arg3toi = atoi(args[2]);
        if (arg3toi > 0) {
            num_cols = arg3toi;
        }
    }

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

    // create a 2D view of inputs
    Kokkos::View<int**, Kokkos::DefaultExecutionSpace> data_device("data", num_flags, num_cols);
    Kokkos::View<int**>::HostMirror data = Kokkos::create_mirror_view(data_device);

    // create a view of flags
    Kokkos::View<int*, Kokkos::DefaultExecutionSpace> flags_device("flags", num_flags);
    Kokkos::View<int*>::HostMirror flags = Kokkos::create_mirror_view(flags_device);

    //! [Setting Up Data]

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Filter And Extract Data");

    //! [Filtering And Extracting Data]
    // create arbitrary data
    for (int i=0; i<num_flags; i++) {
        for (int j=0; j<num_cols; j++) {
            if ((i % 2) == 0) {
                data(i, j) = 1;
            } else {
                data(i, j) = 0;
            }
        }
    }
    // copy the data from host to device
    Kokkos::deep_copy(data_device, data);

    // create arbitrary flags
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
    auto extracted_data = Extract::extractViewByIndex<Kokkos::HostSpace>(data_device, filtered_flags);

    //! [Filtering Data]

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Check Filtered And Extracted Data");

    //! [Checking Filtered And Extracted Data]

    if (filtered_flags.extent(0) != (size_t)num_filtered_flags) {
        all_passed = false;
        std::cout << "Failed - number of filtered flags not matched!" << filtered_flags.extent(0) << " " << num_filtered_flags << std::endl;
    }
    for (size_t i=0; i<filtered_flags.extent(0); i++) {
        if (filtered_flags(i) % 2 != 0) {
            all_passed = false;
            std::cout << "Failed - incorrect filtered flags " << filtered_flags(i) << std::endl;
        }
    }
    // All values inside extracted data should now be 1
    for (size_t i=0; i<extracted_data.extent(0); i++) {
        for (size_t j=0; j<extracted_data.extent(1); j++) {
            if (extracted_data(i, j) != 1) {
                all_passed = false;
                std::cout << "Failed - incorrect values in extracted view at index " << i << " " << j << " " << extracted_data(i, j) << std::endl;
            }
        }
    }

    //! [Checking Filtered And Extracted Data]
    // stop timing comparison loop
    Kokkos::Profiling::popRegion();
    //! [Finalize Program]
    


    //KOKKOS_INLINE_FUNCTION
    //void MyFunctor::operator()(const Test&, const int i) const {
    //    vals(i) += 1;
    //}
    PointData<Kokkos::View<double**> > bah1("label", 100, 1);
    bah1.resumeFillOnHost();

    //const PointData<Kokkos::View<double**> >* bah2 = &bah1;

    MyFunctor myf(&bah1);
    Kokkos::parallel_for("blah", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,MyFunctor::Test>(0,100), myf);
    Kokkos::fence();

    bah1.fillCompleteOnHost();

    for (int i=0; i<100; ++i) {

        printf("val(%d): %f\n", i, bah1.getValueOnHost(i,0));
        //printf("val(%d): %f\n", i, bah1.getValueOnHost(i,0)myf.vals(i));

    }


    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> source_coords_device("source coords", num_flags, 3);
    Kokkos::View<double**>::HostMirror source_coords = Kokkos::create_mirror_view(source_coords_device);
    // Test that build works for device/mirror combinations
    printf("pd1: same type\n");
    PointData<decltype(source_coords_device)> pd1(source_coords_device);
    //printf("pd2: host -> device\n");
    //PointData<decltype(source_coords_device)> pd2(source_coords);
    printf("pd3: hostmirror -> hostmirror\n");
    PointData<decltype(source_coords)> pd3(source_coords);
    //printf("pd4: device to hostmirror.\n");
    //PointData<decltype(source_coords)> pd4(source_coords_device);
    printf("pd5: copy const mirror->mirror (same).\n");
    PointData<decltype(source_coords)> pd5(pd3);
    //printf("pd6: copy const mirror->device.\n");
    //PointData<decltype(source_coords_device)> pd6(pd3);
    //printf("pd7: copy const device->mirror.\n");
    //PointData<decltype(source_coords)> pd7(pd2);
    printf("pd8: assignment mirror->device.\n");
    pd1.deep_copy(pd3);
    //printf("pd9: assignment device->mirror.\n");
    //pd4.deep_copy(pd2);
    //printf("pd10: assignment device->device.\n");
    //pd2.deep_copy(pd1);
    //printf("pd11: assignment mirror->mirror.\n");
    //pd4.deep_copy(pd3);

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
