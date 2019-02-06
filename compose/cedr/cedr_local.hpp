// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_CEDR_LOCAL_HPP
#define INCLUDE_CEDR_LOCAL_HPP

#include "cedr.hpp"
#include "cedr_kokkos.hpp"

namespace cedr {
namespace local {

// Solve
//     min_x sum_i w(i) (x(i) - y(i))^2
//      st   a' x = b
//           xlo <= x <= xhi,
// a(i), w(i) > 0. Return 0 on success and x == y, 1 on success and x != y, -1
// if infeasible, -2 if max_its hit with no solution. See Section 3 of Bochev,
// Ridzal, Shashkov, Fast optimization-based conservative remap of scalar fields
// through aggregate mass transfer. lambda is used in check_1eq_bc_qp_foc.
KOKKOS_INLINE_FUNCTION
Int solve_1eq_bc_qp(const Int n, const Real* w, const Real* a, const Real b,
                    const Real* xlo, const Real* xhi,
                    const Real* y, Real* x, const Int max_its = 100);

KOKKOS_INLINE_FUNCTION
Int solve_1eq_bc_qp_2d(const Real* w, const Real* a, const Real b,
                       const Real* xlo, const Real* xhi,
                       const Real* y, Real* x);

// ClipAndAssuredSum. Does not check for feasibility.
KOKKOS_INLINE_FUNCTION
void caas(const Int n, const Real* a, const Real b,
          const Real* xlo, const Real* xhi,
          const Real* y, Real* x);

Int unittest();

}
}

#include "cedr_local_inl.hpp"

#endif
