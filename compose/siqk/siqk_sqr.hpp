// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_SIQK_SQR_HPP
#define INCLUDE_SIQK_SQR_HPP

#include "siqk_defs.hpp"
#include "siqk_intersect.hpp"

namespace siqk {
namespace sqr { // spherical quadrilateral <-> reference square
/* Let p be a 3x4 matrix with p(:,i) the i'th vertex in a spherical quad in CCW
   order. Let (a,b) be coordinates in the reference square [0,1]^2. (Here we
   choose [0,1] instead of [-1,1].) (a,b) = (0,0) corresponds to p(:,1); (1,0)
   is p(:,2); (1,1) is p(:,3); (0,1) is p(:,4).
     The map from reference square to bilinear quad can be written
       T = p*[ 1 -1 1 -1
              -1  1 0  0
              -1  0 0  1
               1  0 0  0]';
       f(a,b) = T(:,1)*a*b + T(:,2)*a + T(:,3)*b + T(:,4);
   The map to the sphere is then completed with
       g(a,b) = norm(f(a,b))
       q = f(a,b) / g(a,b).
   The Jacobian matrix for q is given by
       q_a = f_a/g - (f g_a)/g^2
       g_a = g_f f_a
   and similarly for q_b.
*/

namespace impl {
// Compute T(i,:).
template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_T_row (const ConstVec3sT& p, const Quad& e, const Int i,
                 Real& t1, Real& t2, Real& t3, Real& t4) {
  t4 = p(e[0],i);
  t3 = -t4 + p(e[3],i);
  t2 = -t4 + p(e[1],i);
  t1 = -t2 + p(e[2],i) - p(e[3],i);
}

// Compute T(:,1)*a*b + T(:,2)*a + T(:,3)*b + T(:,4).
template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_ref_to_bilinear (const ConstVec3sT& p, const Quad& e,
                           Real a, Real b, Real q[3]) {
  a = 0.5*(a + 1);
  b = 0.5*(b + 1);
  for (Int i = 0; i < 3; ++i) {
    Real t1, t2, t3, t4;
    impl::calc_T_row(p, e, i, t1, t2, t3, t4);
    q[i] = t1*a*b + t2*a + t3*b + t4;
  }
}

// The residual function is r(a,b) = f(a,b)/g(a,b) - q.
template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_residual (const ConstVec3sT& p, const Quad& e, const Real a,
                    const Real b, const Real q[3], Real r[3]) {
  calc_ref_to_bilinear(p, e, a, b, r);
  const Real rnorm = std::sqrt(SphereGeometry::norm2(r));
  for (Int i = 0; i < 3; ++i)
    r[i] = r[i]/rnorm - q[i];  
}

// Compute the Jacobian matrix of the residual function: Jacobian(ref square ->
// sphere).
//   TODO Consider rewriting this in terms of the p=1 basis isoparametric
// interpolation formulation. Better performance? See
// calc_isoparametric_jacobian in slmmir.cpp.
template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_Jacobian (const ConstVec3sT& p, const Quad& e, Real a, Real b,
                    Real J[6]) {
  a = 0.5*(a + 1);
  b = 0.5*(b + 1);  
  Real r[3];
  for (Int i = 0; i < 3; ++i) {
    Real t1, t2, t3, t4;
    calc_T_row(p, e, i, t1, t2, t3, t4);
    r[  i] = t1*a*b + t2*a + t3*b + t4;
    J[  i] = t1*b + t2;
    J[3+i] = t1*a + t3;
  }
  Real rtJ[2] = {0};
  for (Int j = 0; j < 2; ++j) {
    const Real* const Jj = J + 3*j;
    for (Int i = 0; i < 3; ++i)
      rtJ[j] += r[i]*Jj[i];
  }
  const Real rnorm2 = SphereGeometry::norm2(r), rnorm = std::sqrt(rnorm2);
  for (Int j = 0; j < 2; ++j) {
    Real* const Jj = J + 3*j;
    for (Int i = 0; i < 3; ++i)
      Jj[i] = (Jj[i] - r[i]*rtJ[j]/rnorm2)/rnorm;
  }
}

// Solve J dx = r.
KOKKOS_INLINE_FUNCTION
void solve_Jxr (Real J[6], const Real r[3], Real dx[2]) {
  // QR factorization: J -> J [n1 a; 0 n2].
  const Real n1 = std::sqrt(SphereGeometry::norm2(J));
  SphereGeometry::scale(1/n1, J);
  const Real a = SphereGeometry::dot(J, J+3);
  SphereGeometry::axpy(-a, J, J+3);
  const Real n2 = std::sqrt(SphereGeometry::norm2(J+3));
  SphereGeometry::scale(1/n2, J+3);
  // r -> Q' r.
  Real Qtr[2] = {0};
  for (Int j = 0; j < 2; ++j) {
    const Real* const Jj = J + 3*j;
    for (Int i = 0; i < 3; ++i)
      Qtr[j] += Jj[i]*r[i];
  }
  // dx = R \ (Q' r).
  dx[1] = 2*(Qtr[1] / n2);
  dx[0] = 2*((Qtr[0] - a*dx[1]) / n1);
}
} // namespace impl

struct Info {
  bool success;
  Int n_iterations;
};

template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_ref_to_sphere (
  // The spherical quad containing the point.
  const ConstVec3sT& p, const Quad& e,
  // (a,b) in [-1,1]
  const Real a, const Real b,
  // The point on the sphere.
  Real q[3])
{
  impl::calc_ref_to_bilinear(p, e, a, b, q);
  SphereGeometry::normalize(q);
}

template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_sphere_to_ref (
  // The spherical quad containing the point.
  const ConstVec3sT& p, const Quad& e,
  // The point on the sphere.
  const Real q[3],
  // (a,b) in [-1,1]
  Real& a, Real& b,
  // Optional info output.
  Info* const info = nullptr,
  // Max number of iterations before returning with failure.
  const Int max_its = 10,
  // Tolerance for Newton iteration.
  const Real tol = 1e2*std::numeric_limits<Real>::epsilon())
{
  const Real tol2 = square(tol);
  Real rnorm2 = 1;
  a = b = 0;
  Int it = 0;
  for (it = 1; it <= max_its; ++it) { // Newton's method.
    Real r[3], J[6];
    impl::calc_residual(p, e, a, b, q, r);
    rnorm2 = SphereGeometry::norm2(r);
    if (rnorm2 <= tol2) break;
    impl::calc_Jacobian(p, e, a, b, J);
    Real dx[2];
    impl::solve_Jxr(J, r, dx);
    a -= dx[0];
    b -= dx[1];
  }
  if (info) {
    info->success = rnorm2 <= tol2;
    info->n_iterations = it;
  }
}

// Ref coords, packed (x,y), CCW, starting from (-1,-1).
KOKKOS_INLINE_FUNCTION
const Real* get_ref_vertices () {
  static const Real c[] = {-1, -1, 1, -1, 1, 1, -1, 1};
  return c;
}

namespace test {
struct Info {
  Int sum_nits, max_nits, nfails;
};

class TestSphereToRefKernel {
  const Real a_test[9] = {-0.1, -1e-16, 0, 1e-15, 0.1, 0.7, 1, 1-1e-14, 1.1};
  const Int n_a_test = sizeof(a_test)/sizeof(*a_test);

  const Real tol_;
  mutable ConstVec3s p_;
  mutable ConstIdxs e_;

public:
  typedef Info value_type;

  TestSphereToRefKernel (const ConstVec3s::HostMirror& p_hm,
                         const ConstIdxs::HostMirror& e_hm,
                         const Real tol = 1e1*std::numeric_limits<Real>::epsilon())
    : tol_(tol)
  {
    { Vec3s p; resize_and_copy(p, p_hm); p_ = p; }
    { Idxs e; resize_and_copy(e, e_hm); e_ = e; }
  }

  Int n () const { return nslices(e_)*square(n_a_test); }
  const Real& tol () const { return tol_; }

  KOKKOS_INLINE_FUNCTION
  void operator() (const Int k, value_type& jinfo) const {
    const Int
      ei = k / square(n_a_test),
      ij = k % square(n_a_test),
      i = ij / n_a_test,
      j = ij % n_a_test;
    const Real a_t = a_test[i], b_t = a_test[j];
    Real q[3];
    sqr::calc_ref_to_sphere(p_, slice(e_, ei), a_t, b_t, q);
    Real a, b;
    sqr::Info info;
    sqr::calc_sphere_to_ref(p_, slice(e_, ei), q, a, b, &info, 100, tol_);
    const Real err = std::sqrt(square(a_t - a) + square(b_t - b));
    // tol is on dx, not (a,b), so adjust slightly.
    if ( ! info.success || err > 1e4*tol_) {
      jinfo.nfails++;
      printf("calc_sphere_to_ref ei %d i %d j %d: nits %d re %1.1e\n",
             ei, i, j, info.n_iterations, err);
    }
    jinfo.sum_nits += info.n_iterations;
    jinfo.max_nits = max(jinfo.max_nits, info.n_iterations);
  }

  KOKKOS_INLINE_FUNCTION
  void init (value_type& info) {
    info.sum_nits = 0;
    info.max_nits = 0;
    info.nfails = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type& dst, volatile value_type const& src) const {
    dst.max_nits = max(dst.max_nits, src.max_nits);
    dst.sum_nits += src.sum_nits;
    dst.nfails += src.nfails;
  }
};

inline Int test_sphere_to_ref (const ConstVec3s::HostMirror& p,
                               const ConstIdxs::HostMirror& e) {
  TestSphereToRefKernel k(p, e);
  Info info;
  auto t = tic();
  ko::parallel_reduce(k.n(), k, info);
  const auto et = toc(t);
  fprintf(stderr, "sqr: #fails %d #iterations mean %1.1f max %d\n",
          info.nfails, (Real) info.sum_nits / k.n(), info.max_nits);
  print_times("test_sphere_to_ref", et);
  return info.nfails;
}
} // namespace test
} // namespace sqr
} // namespace siqk

#endif // INCLUDE_SIQK_SQR_HPP
