// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#include "cedr_test.hpp"
#include "cedr_qlt.hpp"
#include "cedr_caas.hpp"

#include <algorithm>

namespace cedr {
namespace test {
namespace transport1d {

namespace interp {
inline Real to_periodic_core (const Real& xl, const Real& xr, const Real& x) {
  if (x >= xl && x <= xr) return x;
  const Real w = xr - xl, xmxl = x - xl;
  return x - w*std::floor(xmxl / w);
}

inline Real get_slope (const Real x[2], const Real y[2]) {
  return (y[1] - y[0]) / (x[1] - x[0]);
}

inline void
get_cubic (Real dx, Real v1, Real s1, Real v2, Real s2, Real c[4]) {
  Real dx2 = dx*dx;
  Real dx3 = dx2*dx;
  Real den = -dx3;
  Real b1, b2;
  c[2] = s1;
  c[3] = v1;
  b1 = v2 - dx*c[2] - c[3];
  b2 = s2 - c[2];
  c[0] = (2.0*b1 - dx*b2) / den;
  c[1] = (-3.0*dx*b1 + dx2*b2) / den;
}

void cubic_interp_periodic (
  const Real* const x, const Int nx, const Real* const y,
  const Real* const xi, const Int nxi, Real* const yi,
  Int* const dod)
{
  const int nc = nx - 1;
#ifdef _OPENMP
# pragma omp parallel for
#endif
  for (Int j = 0; j < nxi; ++j) {
    const Real xi_per = to_periodic_core(x[0], x[nc], xi[j]);
    Int ip1 = std::upper_bound(x, x + nx, xi_per) - x;
    // Handle numerical issues at boundaries.
    if (ip1 == 0) ++ip1;
    else if (ip1 == nx) --ip1;
    const Int i = ip1 - 1;
    // Domain of dependence.
    Int* dodj = dod + 4*j;
    for (Int k = 0; k < 4; ++k)
      dodj[k] = (i - 1 + k + nc) % nc;
    // Slopes.
    const bool at_start = i == 0, at_end = i == nc - 1;
    const Real smid = get_slope(x+i, y+i);
    Real s1, s2;
    if (at_start) {
      const Real a = (x[nc] - x[nc-1]) / ((x[1] - x[0]) + (x[nc] - x[nc-1]));
      s1 = (1 - a)*get_slope(x+nc-1, y+nc-1) + a*smid;
    } else {
      const Real a = (x[i] - x[i-1]) / (x[ip1] - x[i-1]);
      s1 = (1 - a)*get_slope(x+i-1, y+i-1) + a*smid;
    }
    if (at_end) {
      const Real a = (x[ip1] - x[i]) / ((x[ip1] - x[i]) + (x[1] - x[0]));
      s2 = (1 - a)*smid + a*get_slope(x, y);
    } else {
      const Real a = (x[ip1] - x[i]) / (x[i+2] - x[i]);
      s2 = (1 - a)*smid + a*get_slope(x+ip1, y+ip1);
    }
    // Interp.
    Real c[4];
    get_cubic(x[ip1] - x[i], y[i], s1, y[ip1], s2, c);
    const Real xij = xi_per - x[i];
    yi[j] = (((c[0]*xij + c[1])*xij) + c[2])*xij + c[3];
  }
}
} // namespace interp

class PyWriter {
  typedef std::unique_ptr<FILE, cedr::util::FILECloser> FilePtr;
  FilePtr fh_;
public:
  PyWriter(const std::string& filename);
  void write(const std::string& field_name, const std::vector<Real>& v) const;
};

PyWriter::PyWriter (const std::string& filename) {
  fh_ = FilePtr(fopen((filename + ".py").c_str(), "w"));
  fprintf(fh_.get(), "s = {};\n");
}

void PyWriter::write (const std::string& field_name, const std::vector<Real>& v) const {
  fprintf(fh_.get(), "s['%s'] = [", field_name.c_str());
  for (const auto& e: v)
    fprintf(fh_.get(), " %1.15e,", e);
  fprintf(fh_.get(), "]\n");
}

struct InitialCondition {
  enum Enum { sin, bell, rect, uniform };
  static std::string convert (const Enum& e) {
    switch (e) {
    case Enum::sin: return "sin";
    case Enum::bell: return "bell";
    case Enum::rect: return "rect";
    case Enum::uniform: return "uniform";
    }
    cedr_throw_if(true, "InitialCondition::convert can't convert " << e);
  }
  static Enum convert (const std::string& s) {
    using util::eq;
    if (eq(s, "sin")) return Enum::sin;
    if (eq(s, "bell")) return Enum::bell;
    if (eq(s, "rect")) return Enum::rect;
    if (eq(s, "uniform")) return Enum::uniform;
    cedr_throw_if(true, "InitialCondition::convert can't convert " << s);
  }
  static Real eval (const Enum& ic, const Real x) {
    switch (ic) {
    case Enum::sin: return 0.1 + 0.8*0.5*(1 + std::sin(6*M_PI*x));
    case Enum::bell: return x < 0.5 ? std::sin(2*M_PI*x) : 0;
    case Enum::rect: return x > 0.66 || x < 0.33 ? 0 : 1;
    case Enum::uniform: return 0.42;
    }
    cedr_throw_if(true, "InitialCondition::eval can't convert " << ic);
  }
};

class Problem1D {
  std::vector<Real> xb_, xcp_, rwrk_;
  std::vector<Int> iwrk_;

  void init_mesh (const Int ncells, const bool nonuniform_mesh) {
    xb_.resize(ncells+1);
    xcp_.resize(ncells+1);
    xb_[0] = 0;
    if (nonuniform_mesh) {
      // Large-scale, continuous variation in cell size, plus a huge jump at the
      // periodic boundary.
      for (Int i = 1; i <= ncells; ++i) {
        const Real x = cedr::util::square(Real(i) / ncells);
        xb_[i] = 0.01 + sin(0.5*M_PI*x*x*x*x);
      }
      // Random local cell sizes.
      for (Int i = 1; i <= ncells; ++i)
        xb_[i] *= 0.3 + cedr::util::urand();
      // Cumsum.
      for (Int i = 1; i <= ncells; ++i)
        xb_[i] += xb_[i-1];
      // Normalize.
      for (Int i = 1; i <= ncells; ++i)
        xb_[i] /= xb_[ncells];
    } else {
      xb_.back() = 1;
      for (Int i = 1; i < ncells; ++i)
        xb_[i] = Real(i) / ncells;
    }
    for (Int i = 0; i < ncells; ++i)
      xcp_[i] = 0.5*(xb_[i] + xb_[i+1]);
    xcp_.back() = 1 + xcp_[0];
  }

  static void run_cdr (const Problem1D& p, CDR& cdr,
                       const Real* yp, Real* y, const Int* dods) {
    const Int n = p.ncells();
    for (Int i = 0; i < n; ++i) {
      const Int* dod = dods + 4*i;
      Real min = yp[dod[0]], max = min;
      for (Int j = 1; j < 4; ++j) {
        const Real v = yp[dod[j]];
        min = std::min(min, v);
        max = std::max(max, v);
      }
      const Real area_i = p.area(i);
      cdr.set_Qm(i, 0, y[i]*area_i, min*area_i, max*area_i, yp[i]*area_i);
    }
    cdr.run();
    for (Int i = 0; i < n; ++i)
      y[i] = cdr.get_Qm(i, 0) / p.area(i);
    y[n] = y[0];
  }

  static void run_caas (const Problem1D& p, const Real* yp, Real* y, const Int* dods) {
    const Int n = p.ncells();
    std::vector<Real> lo(n), up(n), w(n);
    Real m = 0;
    for (Int i = 0; i < n; ++i) {
      const Int* dod = dods + 4*i;
      Real min = yp[dod[0]], max = min;
      for (Int j = 1; j < 4; ++j) {
        const Real v = yp[dod[j]];
        min = std::min(min, v);
        max = std::max(max, v);
      }
      const Real area_i = p.area(i);
      lo[i] = min*area_i;
      up[i] = max*area_i;
      y[i] = std::max(min, std::min(max, y[i]));
      m += (yp[i] - y[i])*area_i;
    }
    Real wsum = 0;
    for (Int i = 0; i < n; ++i) {
      w[i] = m >= 0 ? up[i] - y[i]*p.area(i) : y[i]*p.area(i) - lo[i];
      wsum += w[i];
    }
    for (Int i = 0; i < n; ++i)
      y[i] += (m/(wsum*p.area(i)))*w[i];
  }

public:
  Problem1D (const Int ncells, const bool nonuniform_mesh = false) {
    init_mesh(ncells, nonuniform_mesh);
  }

  Int ncells () const { return xb_.size() - 1; }
  Real xb (const Int& i) const { return xb_[i]; }
  Real xcp (const Int& i) const { return xcp_[i]; }
  Real area (const Int& i) const { return xb_[i+1] - xb_[i]; }

  const std::vector<Real> get_xb () const { return xb_; }
  const std::vector<Real> get_xcp () const { return xcp_; }

  void cycle (const Int& nsteps, const Real* y0, Real* yf, CDR* cdr = nullptr) {
    const Int n = xcp_.size();
    rwrk_.resize(2*n);
    iwrk_.resize(4*n);
    Real* xcpi = rwrk_.data();
    Int* dod = iwrk_.data();

    const Real xos = -1.0 / nsteps;
    for (Int i = 0; i < n; ++i)
      xcpi[i] = xcp_[i] + xos;

    Real* ys[] = {xcpi + n, yf};
    std::copy(y0, y0 + n, ys[0]);
    for (Int ti = 0; ti < nsteps; ++ti) {
      interp::cubic_interp_periodic(xcp_.data(), n, ys[0],
                                    xcpi, n, ys[1], dod);
      if (cdr)
        run_cdr(*this, *cdr, ys[0], ys[1], dod);
      else
        run_caas(*this, ys[0], ys[1], dod);
      std::swap(ys[0], ys[1]);
    }
    std::copy(ys[0], ys[0] + n, yf);
  }
};

//todo Clean this up. Right now everything is hardcoded and kludgy.
// - optional write
// - some sort of brief quantitative output
// - better, more canonical IC
// - optional tree imbalance
// - optional mesh nonuniformity
// - parallel?
Int run (const mpi::Parallel::Ptr& parallel, const Input& in) {
  cedr_throw_if(parallel->size() > 1, "run_1d_transport_test runs in serial only.");
  Int nerr = 0;

  Problem1D p(in.ncells, false /* nonuniform_mesh */ );

  auto tree = qlt::tree::make_tree_over_1d_mesh(parallel, in.ncells,
                                                false /* imbalanced */);
  typedef qlt::QLT<Kokkos::DefaultHostExecutionSpace> QLTT;
  QLTT qltnn(parallel, in.ncells, tree), qlt(parallel, in.ncells, tree);

  typedef caas::CAAS<Kokkos::DefaultHostExecutionSpace> CAAST;
  CAAST caas(parallel, in.ncells);

  CDR* cdrs[] = {&qltnn, &qlt, &caas};
  const int ncdrs = sizeof(cdrs)/sizeof(*cdrs);

  bool first = true;
  for (CDR* cdr : cdrs) {
    if (first) {
      QLTT* qlt = dynamic_cast<QLTT*>(cdr);
      cedr_assert(qlt);
      qlt->declare_tracer(cedr::ProblemType::conserve |
                          cedr::ProblemType::nonnegative, 0);
      first = false;
    } else
      cdr->declare_tracer(cedr::ProblemType::conserve |
                          cedr::ProblemType::shapepreserve, 0);
    cdr->end_tracer_declarations();
    for (Int i = 0; i < in.ncells; ++i)
      cdr->set_rhom(i, 0, p.area(i));
    cdr->print(std::cout);
  }

  std::vector<Real> y0(in.ncells+1);
  for (Int i = 0, nc = p.ncells(); i < nc; ++i)
    y0[i] = (p.xcp(i) < 0.4 || p.xcp(i) > 0.9 ?
             InitialCondition::eval(InitialCondition::sin, p.xcp(i)) :
             InitialCondition::eval(InitialCondition::rect, p.xcp(i)));
  y0.back() = y0[0];

  PyWriter w("out_transport1d");
  w.write("xb", p.get_xb());
  w.write("xcp", p.get_xcp());
  w.write("y0", y0);

  std::vector<Real> yf(in.ncells+1);
  const Int nsteps = Int(3.17*in.ncells);
  const Int ncycles = 1;
  
  const char* names[] = {"yqltnn", "yqlt", "ycaas"};
  for (int ic = 0; ic < ncdrs; ++ic) {
    std::copy(y0.begin(), y0.end(), yf.begin());
    for (Int i = 0; i < ncycles; ++i)
      p.cycle(nsteps, yf.data(), yf.data(), cdrs[ic]);
    w.write(names[ic], yf);
  }

  std::copy(y0.begin(), y0.end(), yf.begin());
  for (Int i = 0; i < ncycles; ++i)
    p.cycle(nsteps, yf.data(), yf.data());
  w.write("ylcaas", yf);

  return nerr;
}

} // namespace transport1d
} // namespace test
} // namespace cedr
