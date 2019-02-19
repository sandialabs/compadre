// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#include "cedr_caas.hpp"
#include "cedr_util.hpp"
#include "cedr_test_randomized.hpp"

namespace Kokkos {
struct Real2 {
  cedr::Real v[2];
  KOKKOS_INLINE_FUNCTION Real2 () { v[0] = v[1] = 0; }

  KOKKOS_INLINE_FUNCTION void operator= (const Real2& s) {
    v[0] = s.v[0];
    v[1] = s.v[1];
  }
  KOKKOS_INLINE_FUNCTION void operator= (const volatile Real2& s) volatile {
    v[0] = s.v[0];
    v[1] = s.v[1];
  }

  KOKKOS_INLINE_FUNCTION Real2& operator+= (const Real2& o) {
    v[0] += o.v[0];
    v[1] += o.v[1];
    return *this;
  }
};

template<> struct reduction_identity<Real2> {
  KOKKOS_INLINE_FUNCTION static Real2 sum() { return Real2(); }
};
} // namespace Kokkos

namespace cedr {
namespace caas {

template <typename ES>
CAAS<ES>::CAAS (const mpi::Parallel::Ptr& p, const Int nlclcells,
                const typename UserAllReducer::Ptr& uar)
  : p_(p), user_reducer_(uar), nlclcells_(nlclcells), nrhomidxs_(0),
    need_conserve_(false)
{
  cedr_throw_if(nlclcells == 0, "CAAS does not support 0 cells on a rank.");
  tracer_decls_ = std::make_shared<std::vector<Decl> >();  
}

template <typename ES>
void CAAS<ES>::declare_tracer(int problem_type, const Int& rhomidx) {
  cedr_throw_if( ! (problem_type & ProblemType::shapepreserve),
                "CAAS does not support ! shapepreserve yet.");
  cedr_throw_if(rhomidx > 0, "rhomidx > 0 is not supported yet.");
  tracer_decls_->push_back(Decl(problem_type, rhomidx));
  if (problem_type & ProblemType::conserve)
    need_conserve_ = true;
  nrhomidxs_ = std::max(nrhomidxs_, rhomidx+1);
}

template <typename ES>
void CAAS<ES>::end_tracer_declarations () {
  cedr_throw_if(tracer_decls_->size() == 0, "#tracers is 0.");
  cedr_throw_if(nrhomidxs_ == 0, "#rhomidxs is 0.");
  probs_ = IntList("CAAS probs", static_cast<Int>(tracer_decls_->size()));
  probs_h_ = Kokkos::create_mirror_view(probs_);
  //t2r_ = IntList("CAAS t2r", static_cast<Int>(tracer_decls_->size()));
  for (Int i = 0; i < probs_.extent_int(0); ++i) {
    probs_h_(i) = (*tracer_decls_)[i].probtype;
    //t2r_(i) = (*tracer_decls_)[i].rhomidx;
  }
  Kokkos::deep_copy(probs_, probs_h_);
  tracer_decls_ = nullptr;
  // (rho, Qm, Qm_min, Qm_max, [Qm_prev])
  const Int e = need_conserve_ ? 1 : 0;
  d_ = RealList("CAAS data", nlclcells_ * ((3+e)*probs_.size() + 1));
  const auto nslots = 4*probs_.size();
  // (e'Qm_clip, e'Qm, e'Qm_min, e'Qm_max, [e'Qm_prev])
  send_ = RealList("CAAS send", nslots*(user_reducer_ ? nlclcells_ : 1));
  recv_ = RealList("CAAS recv", nslots);
}

template <typename ES>
int CAAS<ES>::get_problem_type (const Int& tracer_idx) const {
  cedr_assert(tracer_idx >= 0 && tracer_idx < probs_.extent_int(0));
  return probs_h_[tracer_idx];
}

template <typename ES>
Int CAAS<ES>::get_num_tracers () const {
  return probs_.extent_int(0);
}

template <typename RealList, typename IntList>
KOKKOS_INLINE_FUNCTION static void
calc_Qm_scalars (const RealList& d, const IntList& probs,
                 const Int& nt, const Int& nlclcells,
                 const Int& k, const Int& os, const Int& i,
                 Real& Qm_clip, Real& Qm_term) {
  const Real Qm = d(os+i);
  Qm_term = (probs(k) & ProblemType::conserve ?
             d(os + nlclcells*3*nt + i) /* Qm_prev */ :
             Qm);
  const Real Qm_min = d(os + nlclcells*  nt + i);
  const Real Qm_max = d(os + nlclcells*2*nt + i);
  Qm_clip = cedr::impl::min(Qm_max, cedr::impl::max(Qm_min, Qm));
}

template <typename ES>
void CAAS<ES>::reduce_locally () {
  const bool user_reduces = user_reducer_ != nullptr;
  ConstExceptGnu Int nt = probs_.size(), nlclcells = nlclcells_;

  const auto probs = probs_;
  const auto send = send_;
  const auto d = d_;
  if (user_reduces) {
    const auto calc_Qm_clip = KOKKOS_LAMBDA (const Int& j) {
      const auto k = j / nlclcells;
      const auto i = j % nlclcells;
      const auto os = (k+1)*nlclcells;
      Real Qm_clip, Qm_term;
      calc_Qm_scalars(d, probs, nt, nlclcells, k, os, i, Qm_clip, Qm_term);
      d(os+i) = Qm_clip;
      send(nlclcells*      k  + i) = Qm_clip;
      send(nlclcells*(nt + k) + i) = Qm_term;
    };
    Kokkos::parallel_for(Kokkos::RangePolicy<ES>(0, nt*nlclcells), calc_Qm_clip);
    const auto set_Qm_minmax = KOKKOS_LAMBDA (const Int& j) {
      const auto k = 2*nt + j / nlclcells;
      const auto i = j % nlclcells;
      const auto os = (k-nt+1)*nlclcells;
      send(nlclcells*k + i) = d(os+i);
    };
    Kokkos::parallel_for(Kokkos::RangePolicy<ES>(0, 2*nt*nlclcells), set_Qm_minmax);
  } else {
    using ESU = cedr::impl::ExeSpaceUtils<ES>;
    const auto calc_Qm_clip = KOKKOS_LAMBDA (const typename ESU::Member& t) {
      const auto k = t.league_rank();
      const auto os = (k+1)*nlclcells;
      const auto reduce = [&] (const Int& i, Kokkos::Real2& accum) {
        Real Qm_clip, Qm_term;
        calc_Qm_scalars(d, probs, nt, nlclcells, k, os, i, Qm_clip, Qm_term);
        d(os+i) = Qm_clip;
        accum.v[0] += Qm_clip;
        accum.v[1] += Qm_term;
      };
      Kokkos::Real2 accum;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, nlclcells),
                              reduce, Kokkos::Sum<Kokkos::Real2>(accum));
      send(     k) = accum.v[0];
      send(nt + k) = accum.v[1];
    };
    Kokkos::parallel_for(ESU::get_default_team_policy(nt, nlclcells),
                         calc_Qm_clip);
    const auto set_Qm_minmax = KOKKOS_LAMBDA (const typename ESU::Member& t) {
      const auto k = 2*nt + t.league_rank();
      const auto os = (k-nt+1)*nlclcells;
      Real accum = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, nlclcells),
                              [&] (const Int& i, Real& accum) { accum += d(os+i); },
                              Kokkos::Sum<Real>(accum));
      send(k) = accum;
    };
    Kokkos::parallel_for(ESU::get_default_team_policy(2*nt, nlclcells),
                         set_Qm_minmax);
  }
}

template <typename ES>
void CAAS<ES>::reduce_globally () {
  const int err = mpi::all_reduce(*p_, send_.data(), recv_.data(),
                                  send_.size(), MPI_SUM);
  cedr_throw_if(err != MPI_SUCCESS,
                "CAAS::reduce_globally MPI_Allreduce returned " << err);
}

template <typename ES>
void CAAS<ES>::finish_locally () {
  using ESU = cedr::impl::ExeSpaceUtils<ES>;
  ConstExceptGnu Int nt = probs_.size(), nlclcells = nlclcells_;
  const auto recv = recv_;
  const auto d = d_;
  const auto adjust_Qm = KOKKOS_LAMBDA (const typename ESU::Member& t) {
    const auto k = t.league_rank();
    const auto os = (k+1)*nlclcells;
    const auto Qm_clip_sum = recv(     k);
    const auto Qm_sum      = recv(nt + k);
    const auto m = Qm_sum - Qm_clip_sum;
    if (m < 0) {
      const auto Qm_min_sum = recv(2*nt + k);
      auto fac = Qm_clip_sum - Qm_min_sum;
      if (fac > 0) {
        fac = m/fac;
        const auto adjust = [&] (const Int& i) {
          const auto Qm_min = d(os + nlclcells * nt + i);
          auto& Qm = d(os+i);
          Qm += fac*(Qm - Qm_min);
          Qm = impl::max(Qm_min, Qm);
        };
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nlclcells), adjust);
      }
    } else if (m > 0) {
      const auto Qm_max_sum = recv(3*nt + k);
      auto fac = Qm_max_sum - Qm_clip_sum;
      if (fac > 0) {
        fac = m/fac;
        const auto adjust = [&] (const Int& i) {
          const auto Qm_max = d(os + nlclcells*2*nt + i);
          auto& Qm = d(os+i);
          Qm += fac*(Qm_max - Qm);
          Qm = impl::min(Qm_max, Qm);
        };
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nlclcells), adjust);
      }
    }
  };
  Kokkos::parallel_for(ESU::get_default_team_policy(nt, nlclcells),
                       adjust_Qm);
}

template <typename ES>
void CAAS<ES>::run () {
  reduce_locally();
  const bool user_reduces = user_reducer_ != nullptr;
  if (user_reduces)
    (*user_reducer_)(*p_, send_.data(), recv_.data(),
                     nlclcells_, recv_.size(), MPI_SUM);
  else
    reduce_globally();
  finish_locally();
}

namespace test {
struct TestCAAS : public cedr::test::TestRandomized {
  typedef CAAS<Kokkos::DefaultExecutionSpace> CAAST;

  struct TestAllReducer : public CAAST::UserAllReducer {
    int operator() (const mpi::Parallel& p, Real* sendbuf, Real* rcvbuf,
                    int nlcl, int count, MPI_Op op) const override {
      Kokkos::View<Real*> s(sendbuf, nlcl*count), r(rcvbuf, count);
      const auto s_h = Kokkos::create_mirror_view(s);
      Kokkos::deep_copy(s_h, s);
      const auto r_h = Kokkos::create_mirror_view(r);
      for (int i = 1; i < nlcl; ++i)
        s_h(0) += s_h(i);
      for (int k = 1; k < count; ++k) {
        s_h(k) = s_h(nlcl*k);
        for (int i = 1; i < nlcl; ++i)
          s_h(k) += s_h(nlcl*k + i);
      }
      const int err = mpi::all_reduce(p, s_h.data(), r_h.data(), count, op);
      Kokkos::deep_copy(r, r_h);
      return err;
    }
  };

  TestCAAS (const mpi::Parallel::Ptr& p, const Int& ncells,
            const bool use_own_reducer, const bool verbose)
    : TestRandomized("CAAS", p, ncells, verbose),
      p_(p)
  {
    const auto np = p->size(), rank = p->rank();
    nlclcells_ = ncells / np;
    const Int todo = ncells - nlclcells_ * np;
    if (rank < todo) ++nlclcells_;
    caas_ = std::make_shared<CAAST>(
      p, nlclcells_,
      use_own_reducer ? std::make_shared<TestAllReducer>() : nullptr);
    init();
  }

  CDR& get_cdr () override { return *caas_; }

  void init_numbering () override {
    const auto np = p_->size(), rank = p_->rank();
    Int start = 0;
    for (Int lrank = 0; lrank < rank; ++lrank)
      start += get_nllclcells(ncells_, np, lrank);
    gcis_.resize(nlclcells_);
    for (Int i = 0; i < nlclcells_; ++i)
      gcis_[i] = start + i;
  }

  void init_tracers () override {
    // CAAS doesn't yet support everything, so remove a bunch of the tracers.
    std::vector<TestRandomized::Tracer> tracers;
    Int idx = 0;
    for (auto& t : tracers_) {
      if ( ! (t.problem_type & ProblemType::shapepreserve) ||
           ! t.local_should_hold)
        continue;
      t.idx = idx++;
      tracers.push_back(t);
      caas_->declare_tracer(t.problem_type, 0);
    }
    tracers_ = tracers;
    caas_->end_tracer_declarations();
  }

  void run_impl (const Int trial) override {
    caas_->run();
  }

private:
  mpi::Parallel::Ptr p_;
  Int nlclcells_;
  CAAST::Ptr caas_;

  static Int get_nllclcells (const Int& ncells, const Int& np, const Int& rank) {
    Int nlclcells = ncells / np;
    const Int todo = ncells - nlclcells * np;
    if (rank < todo) ++nlclcells;
    return nlclcells;
  }
};

Int unittest (const mpi::Parallel::Ptr& p) {
  const auto np = p->size();
  Int nerr = 0;
  for (Int nlclcells : {1, 2, 4, 11}) {
    Long ncells = np*nlclcells;
    if (ncells > np) ncells -= np/2;
    nerr += TestCAAS(p, ncells, false, false).run<TestCAAS::CAAST>(1, false);
    nerr += TestCAAS(p, ncells, true, false).run<TestCAAS::CAAST>(1, false);
  }
  return nerr;
}
} // namespace test
} // namespace caas
} // namespace cedr

#ifdef KOKKOS_ENABLE_SERIAL
template class cedr::caas::CAAS<Kokkos::Serial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template class cedr::caas::CAAS<Kokkos::OpenMP>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class cedr::caas::CAAS<Kokkos::Cuda>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class cedr::caas::CAAS<Kokkos::Threads>;
#endif
