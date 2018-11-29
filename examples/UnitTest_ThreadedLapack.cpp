#include <GMLS_Config.h>
#include "Compadre_LinearAlgebra_Definitions.hpp"

#include <assert.h>
#include <Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>

#ifdef COMPADRE_USE_MPI
#include <mpi.h>
#endif

using namespace Compadre;

//! This tests whether or not the LAPACK+BLAS combination is safe to have a parallel_for wrapping them
//! If they are not threadsafe, then nothing in this toolkit will work and another library will have
//! be provided that is threadsafe.

// called from command line
int main (int argc, char* args[]) {

// initializes MPI (if available) with command line arguments given
#ifdef COMPADRE_USE_MPI
MPI_Init(&argc, &args);
#endif

int number_wrong = 0;

// initializes Kokkos with command line arguments given
Kokkos::initialize(argc, args);
{

    const int P_rows   = 100;
    const int P_cols   = 100;
    assert((P_rows >= P_cols) && "P must not be underdetermined.");

    const int RHS_rows = P_rows;

    const int num_matrices = 100;

    Kokkos::Profiling::pushRegion("Instantiating Data");
    auto all_P   = Kokkos::View<double*>("P", num_matrices*P_cols*P_rows);
    auto all_RHS = Kokkos::View<double*>("RHS", num_matrices*RHS_rows*RHS_rows);
    Kokkos::Profiling::popRegion();

    Kokkos::parallel_for("Fill Matrices", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA(const int i) {
        Kokkos::View<double**, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
            P(all_P.data() + i*P_cols*P_rows, P_rows, P_cols);

        Kokkos::View<double**, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
            RHS(all_RHS.data() + i*RHS_rows*RHS_rows, RHS_rows, RHS_rows);

        for (int j=0; j<P_cols; ++j) {
            P(j,j) = 1.0;
        }

        for (int j=0; j<RHS_rows; ++j) {
            RHS(j,j) = 1.0;
        }
    });
    Kokkos::fence();


    GMLS_LinearAlgebra::batchQRFactorize(all_P.ptr_on_device(), P_rows, P_cols, all_RHS.ptr_on_device(), RHS_rows, RHS_rows, RHS_rows, P_cols, num_matrices);


    const double tol = 1e-10;
    Kokkos::parallel_reduce("Check Solution", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,num_matrices), KOKKOS_LAMBDA(const int i, int& t_wrong) {
        Kokkos::View<double**, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
            RHS(all_RHS.data() + i*RHS_rows*RHS_rows, RHS_rows, RHS_rows);

        for (int j=0; j<RHS_rows; ++j) {
            if (std::abs(RHS(j,j)-1) > tol) {
                t_wrong++;
            }
            //for (int k=0; k<RHS_rows; ++k) {
            //    if (j==k) {
            //        if (std::abs(RHS(j,k)-1) > tol) {
            //            t_wrong++;
            //        }
            //    } else {
            //        if (std::abs(RHS(j,k)-0) > tol) {
            //            t_wrong++;
            //        }
            //    }
            //}
        }

    }, number_wrong);

} 
// finalize Kokkos
Kokkos::finalize();
#ifdef COMPADRE_USE_MPI
MPI_Finalize();
#endif

if (number_wrong > 0) {
    printf("Incorrect result. LAPACK IS NOT THREADSAFE AND CANNOT BE USED WITH THIS TOOLKIT! Either provide a thread safe LAPACK+BLAS combination or set -DLAPACK_DECLARED_THREADSAFE:BOOL=OFF in CMake and take a MASSIVE performance hit.\n");
    return -1;
}
return 0;
}
