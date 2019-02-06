// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

# bash make_qltcpp.sh
# mpicxx -Wall -pedantic -fopenmp -std=c++11 -I/home/ambradl/lib/kokkos/cpu/include qlt.cpp -L/home/ambradl/lib/kokkos/cpu/lib -lkokkos -ldl
# OMP_PROC_BIND=false OMP_NUM_THREADS=2 mpirun -np 14 ./a.out -t

(for f in cedr_kokkos.hpp cedr.hpp cedr_mpi.hpp cedr_util.hpp cedr_cdr.hpp cedr_qlt.hpp cedr_caas.hpp cedr_caas_inl.hpp cedr_local.hpp cedr_mpi_inl.hpp cedr_local_inl.hpp cedr_qlt_inl.hpp cedr_test_randomized.hpp cedr_test.hpp cedr_util.cpp cedr_local.cpp cedr_mpi.cpp cedr_qlt.cpp cedr_caas.cpp cedr_test_randomized.cpp cedr_test_1d_transport.cpp cedr_test.cpp; do
    echo "//>> $f"
    cat $f
    echo ""
done) > qlt.cpp
sed sV'#include "cedr'V'//#include "cedr'V -i qlt.cpp
sed sV'#include "compose'V'//#include "compose'V -i qlt.cpp
