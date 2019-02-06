// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_CEDR_HPP
#define INCLUDE_CEDR_HPP

#include "cedr_kokkos.hpp"

// Communication-Efficient Constrained Density Reconstructors
namespace cedr {
typedef int Int;
typedef long int Long;
typedef std::size_t Size;
typedef double Real;

// CDRs in general implement
// * tracer mass, Qm, conservation;
// * mixing ratio, q, shape preservation, either local bound preservation or
//   dynamic range preservation; and
// * tracer consistency, which follows from dynamic range preservation or
//   stronger (including local bound preservation) with rhom coming from the
//   dynamics.
//
// One can solve a subset of these.
//   If !conserve, then the CDR does not alter the tracer mass, but it does not
// correct for any failure in mass conservation in the field given to it.
//   If consistent but !shapepreserve, the the CDR solves the dynamic range
// preservation problem rather than the local bound preservation problem.
struct ProblemType {
  enum : Int { conserve = 1, shapepreserve = 1 << 1, consistent = 1 << 2 };
};
}

#endif
