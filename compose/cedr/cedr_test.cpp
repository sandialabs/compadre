// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#include "cedr_qlt.hpp"
#include "cedr_caas.hpp"
#include "cedr_mpi.hpp"
#include "cedr_util.hpp"
#include "cedr_test.hpp"

#include <stdexcept>
#include <sstream>

namespace cedr { 
struct InputParser {
  qlt::test::Input qin;
  test::transport1d::Input tin;

  class ArgAdvancer {
    const int argc_;
    char const* const* argv_;
    int i_;
  public:
    ArgAdvancer (int argc, char** argv) : argc_(argc), argv_(argv), i_(1) {}
    const char* advance () {
      if (i_+1 >= argc_) cedr_throw_if(true, "Command line is missing an argument.");
      return argv_[++i_];
    }
    const char* token () const { return argv_[i_]; }
    void incr () { ++i_; }
    bool more () const { return i_ < argc_; }
  };
  
  InputParser (int argc, char** argv, const qlt::Parallel::Ptr& p) {
    using util::eq;
    qin.unittest = false;
    qin.perftest = false;
    qin.write = false;
    qin.ncells = 0;
    qin.ntracers = 1;
    qin.tracer_type = 0;
    qin.nrepeat = 1;
    qin.pseudorandom = false;
    qin.verbose = false;
    tin.ncells = 0;
    for (ArgAdvancer aa(argc, argv); aa.more(); aa.incr()) {
      const char* token = aa.token();
      if (eq(token, "-t", "--unittest")) qin.unittest = true;
      else if (eq(token, "-pt", "--perftest")) qin.perftest = true;
      else if (eq(token, "-w", "--write")) qin.write = true;
      else if (eq(token, "-nc", "--ncells")) qin.ncells = std::atoi(aa.advance());
      else if (eq(token, "-nt", "--ntracers")) qin.ntracers = std::atoi(aa.advance());
      else if (eq(token, "-tt", "--tracertype")) qin.tracer_type = std::atoi(aa.advance());
      else if (eq(token, "-nr", "--nrepeat")) qin.nrepeat = std::atoi(aa.advance());
      else if (eq(token, "--proc-random")) qin.pseudorandom = true;
      else if (eq(token, "-v", "--verbose")) qin.verbose = true;
      else if (eq(token, "-t1d", "--transport1dtest")) tin.ncells = 1;
      else cedr_throw_if(true, "Invalid token " << token);
    }

    if (tin.ncells) {
      tin.ncells = qin.ncells;
      tin.verbose = qin.verbose;
    }

    cedr_throw_if(qin.tracer_type < 0 || qin.tracer_type >= 4,
                  "Tracer type is out of bounds [0, 3].");
    cedr_throw_if(qin.ntracers < 1, "Number of tracers is < 1.");
  }

  void print (std::ostream& os) const {
    os << "ncells " << qin.ncells
       << " nrepeat " << qin.nrepeat;
    if (qin.pseudorandom) os << " random";
    os << "\n";
  }
};
} // namespace cedr

int main (int argc, char** argv) {
  int nerr = 0, retval = 0;
  MPI_Init(&argc, &argv);
  auto p = cedr::mpi::make_parallel(MPI_COMM_WORLD);
  srand(p->rank());
  Kokkos::initialize(argc, argv);
#if 0
  try
#endif
  {
    cedr::InputParser inp(argc, argv, p);
    if (p->amroot()) inp.print(std::cout);
    if (inp.qin.unittest) {
      nerr += cedr::local::unittest();
      nerr += cedr::caas::test::unittest(p);
    }
    if (inp.qin.unittest || inp.qin.perftest)
      nerr += cedr::qlt::test::run_unit_and_randomized_tests(p, inp.qin);
    if (inp.tin.ncells > 0)
      nerr += cedr::test::transport1d::run(p, inp.tin);
    {
      int gnerr;
      cedr::mpi::all_reduce(*p, &nerr, &gnerr, 1, MPI_SUM);
      retval = gnerr != 0 ? -1 : 0;
      if (p->amroot())
        std::cout << (gnerr != 0 ? "FAIL" : "PASS") << "\n";
    }
  }
#if 0
  catch (const std::exception& e) {
    if (p->amroot())
      std::cerr << e.what();
    retval = -1;
  }
#endif
  Kokkos::finalize();
  if (nerr) prc(nerr);
  MPI_Finalize();
  return retval;
}
