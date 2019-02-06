// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_CEDR_CAAS_HPP
#define INCLUDE_CEDR_CAAS_HPP

#include "cedr_cdr.hpp"

namespace cedr {
// ClipAndAssuredSum.
namespace caas {

template <typename ExeSpace = Kokkos::DefaultExecutionSpace>
class CAAS : public CDR {
public:
  typedef typename cedr::impl::DeviceType<ExeSpace>::type Device;
  typedef CAAS<ExeSpace> Me;
  typedef std::shared_ptr<Me> Ptr;

public:
  struct UserAllReducer {
    typedef std::shared_ptr<const UserAllReducer> Ptr;
    virtual int operator()(const mpi::Parallel& p,
                           // In Fortran, these are formatted as
                           //   sendbuf(nlocal, nfld)
                           //   rcvbuf(nfld)
                           // The implementation is permitted to modify sendbuf.
                           Real* sendbuf, Real* rcvbuf,
                           // nlocal is number of values to reduce in this rank.
                           // nfld is number of fields.
                           int nlocal, int nfld,
                           MPI_Op op) const = 0;
  };

  CAAS(const mpi::Parallel::Ptr& p, const Int nlclcells,
       const typename UserAllReducer::Ptr& r = nullptr);

  void declare_tracer(int problem_type, const Int& rhomidx) override;

  void end_tracer_declarations() override;

  int get_problem_type(const Int& tracer_idx) const override;

  Int get_num_tracers() const override;

  // lclcellidx is trivial; it is the user's index for the cell.
  KOKKOS_INLINE_FUNCTION
  void set_rhom(const Int& lclcellidx, const Int& rhomidx, const Real& rhom) const override;

  KOKKOS_INLINE_FUNCTION
  void set_Qm(const Int& lclcellidx, const Int& tracer_idx,
              const Real& Qm, const Real& Qm_min, const Real& Qm_max,
              const Real Qm_prev = std::numeric_limits<Real>::infinity()) const override;

  void run() override;

  KOKKOS_INLINE_FUNCTION
  Real get_Qm(const Int& lclcellidx, const Int& tracer_idx) const override;

protected:
  typedef Kokkos::View<Real*, Kokkos::LayoutLeft, Device> RealList;
  typedef cedr::impl::Unmanaged<RealList> UnmanagedRealList;
  typedef Kokkos::View<Int*, Kokkos::LayoutLeft, Device> IntList;

  struct Decl {
    int probtype;
    Int rhomidx;
    Decl (const int probtype_, const Int rhomidx_)
      : probtype(probtype_), rhomidx(rhomidx_) {}
  };

  mpi::Parallel::Ptr p_;
  typename UserAllReducer::Ptr user_reducer_;

  Int nlclcells_, nrhomidxs_;
  std::shared_ptr<std::vector<Decl> > tracer_decls_;
  bool need_conserve_;
  IntList probs_, t2r_;
  typename IntList::HostMirror probs_h_;
  RealList d_, send_, recv_;

  void reduce_globally();

PRIVATE_CUDA:
  void reduce_locally();
  void finish_locally();
};

namespace test {
Int unittest(const mpi::Parallel::Ptr& p);
} // namespace test
} // namespace caas
} // namespace cedr

#include "cedr_caas_inl.hpp"

#endif
