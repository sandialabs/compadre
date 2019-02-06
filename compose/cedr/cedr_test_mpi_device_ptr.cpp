// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#include "cedr_mpi.hpp"

// Seg fault if MPI can't handle device pointers. Thus, we'll isolate
// this environment-related problem from true bugs inside COMPOSE.
void test_mpi_device_ptr () {
  using namespace cedr;
  static const int n = 42, tag = 24;
  Kokkos::View<int*> s("s", n), r("r", n);
  const auto p = mpi::make_parallel(MPI_COMM_WORLD);
  mpi::Request req;
  mpi::irecv(*p, r.data(), n, p->root(), tag, &req);
  mpi::isend(*p, s.data(), n, p->root(), tag);
  mpi::waitall(1, &req);
}

int main (int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv); {
    test_mpi_device_ptr();
  } Kokkos::finalize();
  MPI_Finalize();
  // If we didn't seg fault, we passed.
  return 0;
}
