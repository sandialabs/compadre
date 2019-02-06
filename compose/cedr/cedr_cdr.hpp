// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_CEDR_CDR_HPP
#define INCLUDE_CEDR_CDR_HPP

#include "cedr_mpi.hpp"

namespace cedr {
// Constrained Density Reconstructor interface.
struct CDR {
  typedef std::shared_ptr<CDR> Ptr;

  virtual void print(std::ostream& os) const {}

  // Set up QLT tracer metadata. Call declare_tracer in order of the tracer
  // index in the caller's numbering. Once end_tracer_declarations is called, it
  // is an error to call declare_tracer again.
  //   Associate the tracer with a rhom index. In many problems, there will be
  // only one rhom, so rhomidx is always 0.
  //   It is an error to call this function from a parallel region.
  virtual void declare_tracer(int problem_type, const Int& rhomidx) = 0;

  // It is an error to call this function from a parallel region.
  virtual void end_tracer_declarations() = 0;

  virtual int get_problem_type(const Int& tracer_idx) const = 0;

  virtual Int get_num_tracers() const = 0;

  // set_{rhom,Qm}: Set cell values prior to running the QLT algorithm.
  //
  //   Notation:
  //     rho: Total density.
  //       Q: Tracer density.
  //       q: Tracer mixing ratio = Q/rho.
  //      *m: Mass corresponding to the density; results from an integral over a
  //          region, such as a cell.
  //   Some CDRs have a nontrivial local <-> global cell index map. For these
  // CDRs, lclcellidx may be nontrivial. For others, the caller should provide
  // the index into the local cell.
  //
  //   set_rhom must be called before set_Qm.
  KOKKOS_FUNCTION
  virtual void set_rhom(
    const Int& lclcellidx, const Int& rhomidx,
    // Current total mass in this cell.
    const Real& rhom) const = 0;

  KOKKOS_FUNCTION
  virtual void set_Qm(
    const Int& lclcellidx, const Int& tracer_idx,
    // Current tracer mass in this cell.
    const Real& Qm,
    // Minimum and maximum permitted tracer mass in this cell.
    const Real& Qm_min, const Real& Qm_max,
    // If mass conservation is requested, provide the previous Qm, which will be
    // summed to give the desired global mass.
    const Real Qm_prev = std::numeric_limits<Real>::infinity()) const = 0;

  // Run the QLT algorithm with the values set by set_{rho,Q}. It is an error to
  // call this function from a parallel region.
  virtual void run() = 0;

  // Get a cell's tracer mass Qm after the QLT algorithm has run.
  KOKKOS_FUNCTION
  virtual Real get_Qm(const Int& lclcellidx, const Int& tracer_idx) const = 0;
};
} // namespace cedr

#endif
