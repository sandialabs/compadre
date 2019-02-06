// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

// ko=/home/ambradl/lib/kokkos/cpu; mycpp -I$ko/include -L$ko/lib -fopenmp unit_test.cpp -lkokkos -ldl -Wall -pedantic -DSIQK_TIME
// ./a.out -m | grep "mat=1" > foo.m
// >> msik('draw_unit_test0', 'foo');

#include <limits>

#include "siqk.hpp"
using namespace siqk;

#define INSTANTIATE_PLANE

//> Code that will likely be moved to library files.

template <typename CV3s>
void write_matlab (const std::string& name, const CV3s& p) {
  printf("mat=1; %s = [", name.c_str());
  for (Int ip = 0; ip < nslices(p); ++ip)
    printf(" %1.15e %1.15e %1.15e;", p(ip,0), p(ip,1), p(ip,2));
  printf("].';\n");
}

template <typename CV3s, typename CIs>
void write_matlab (const std::string& name, const CV3s& p, const CIs& e) {
  printf("mat=1; %s.p = [", name.c_str());
  for (Int ip = 0; ip < nslices(p); ++ip)
    printf(" %1.15e %1.15e %1.15e;", p(ip,0), p(ip,1), p(ip,2));
  printf("].';\n");
  printf("mat=1; %s.n = [", name.c_str());
  for (Int ie = 0; ie < nslices(e); ++ie)
    printf(" %d %d %d %d;", e(ie,0)+1, e(ie,1)+1, e(ie,2)+1, e(ie,3)+1);
  printf("].';\n");
}

static void make_planar_mesh (Vec3s::HostMirror& p, Idxs::HostMirror& e,
                              const Int n) {
  const Real d = std::sqrt(0.5);
  ko::resize(e, n*n, 4);
  ko::resize(p, (n+1)*(n+1));
  for (Int iy = 0; iy < n+1; ++iy)
    for (Int ix = 0; ix < n+1; ++ix) {
      const auto idx = (n+1)*iy + ix;
      p(idx,0) = 2*(static_cast<Real>(ix)/n - 0.5)*d;
      p(idx,1) = 2*(static_cast<Real>(iy)/n - 0.5)*d;
      p(idx,2) = 0;
    }
  for (Int iy = 0; iy < n; ++iy)
    for (Int ix = 0; ix < n; ++ix) {
      const auto idx = n*iy + ix;
      e(idx,0) = (n+1)*iy + ix;
      e(idx,1) = (n+1)*iy + ix+1;
      e(idx,2) = (n+1)*(iy+1) + ix+1;
      e(idx,3) = (n+1)*(iy+1) + ix;
    }
}

// Row-major R.
inline void form_rotation (const Real axis[3], const Real angle, Real r[9]) {
  const Real nrm = std::sqrt(SphereGeometry::norm2(axis));
  const Real& x = axis[0] / nrm, & y = axis[1] / nrm, & z = axis[2] / nrm,
    & th = angle;
  const Real cth = std::cos(th), sth = std::sin(th), omcth = 1 - cth;
  r[0] = cth + x*x*omcth;
  r[3] = y*x*omcth + z*sth;
  r[6] = z*x*omcth - y*sth;
  r[1] = x*y*omcth - z*sth;
  r[4] = cth + y*y*omcth;
  r[7] = z*y*omcth + x*sth;
  r[2] = x*z*omcth + y*sth;
  r[5] = y*z*omcth - x*sth;
  r[8] = cth + z*z*omcth;
}

template <typename V>
static void rotate (const Real R[9], V p) {
  const Real x = p[0], y = p[1], z = p[2];
  p[0] = R[0]*x + R[1]*y + R[2]*z;
  p[1] = R[3]*x + R[4]*y + R[5]*z;
  p[2] = R[6]*x + R[7]*y + R[8]*z;
}

template <typename V>
static void translate (const Real xlate[3], V p) {
  for (Int i = 0; i < 3; ++i) p[i] += xlate[i];
}

static void transform_planar_mesh (const Real R[9], const Real xlate[3],
                                   Vec3s::HostMirror& p) {
  for (Int i = 0; i < nslices(p); ++i) {
    rotate(R, slice(p, i));
    translate(xlate, slice(p, i));
  }
}

// Remove vertices marked unused and adjust numbering.
static void remove_unused_vertices (Vec3s::HostMirror& p, Idxs::HostMirror& e,
                                    const Real unused) {
  // adjust[i] is the number to subtract from i. Hence if e(ei,0) was originally
  // i, it is adjusted to i - adjust[i].
  std::vector<Int> adjust(nslices(p), 0);
  Int rmcnt = 0;
  for (Int i = 0; i < nslices(p); ++i) {
    if (p(i,0) != unused) continue;
    adjust[i] = 1;
    ++rmcnt;
  }
  // Cumsum.
  for (Int i = 1; i < nslices(p); ++i)
    adjust[i] += adjust[i-1];
  // Adjust e.
  for (Int ei = 0; ei < nslices(e); ++ei)
    for (Int k = 0; k < szslice(e); ++k)
      e(ei,k) -= adjust[e(ei,k)];
  // Remove unused from p.
  Vec3s::HostMirror pc("copy", nslices(p));
  ko::deep_copy(pc, p);
  ko::resize(p, nslices(p) - rmcnt);
  for (Int i = 0, j = 0; i < nslices(pc); ++i) {
    if (pc(i,0) == unused) continue;
    for (Int k = 0; k < szslice(pc); ++k) p(j,k) = pc(i,k);
    ++j;
  }
}

// A very simple cube-sphere mesh with nxn elements per face. At least for now
// I'm not bothering with making the elements well proportioned.
void make_cubesphere_mesh (Vec3s::HostMirror& p, Idxs::HostMirror& e,
                           const Int n) {
  // Transformation of the reference mesh make_planar_mesh to make each of the
  // six faces.
  const Real d = std::sqrt(0.5);
  static Real R[6][9] = {{ 1, 0, 0, 0, 0, 0, 0, 1, 0},  // face 0, -y
                         { 0, 0, 0, 1, 0, 0, 0, 1, 0},  //      1, +x
                         {-1, 0, 0, 0, 0, 0, 0, 1, 0},  //      2, +y
                         { 0, 0, 0,-1, 0, 0, 0, 1, 0},  //      3, -x
                         { 1, 0, 0, 0, 1, 0, 0, 0, 0},  //      4, +z
                         {-1, 0, 0, 0, 1, 0, 0, 0, 0}}; //      5, -z
  static Real xlate[6][3] = {{ 0,-d, 0}, { d, 0, 0}, { 0, d, 0},
                             {-d, 0, 0}, { 0, 0, d}, { 0, 0,-d}};
  // Construct 6 uncoupled faces.
  Vec3s::HostMirror ps[6];
  Vec3s::HostMirror& p_ref = ps[0];
  Idxs::HostMirror es[6];
  Idxs::HostMirror& e_ref = es[0];
  make_planar_mesh(p_ref, e_ref, n);
  ko::resize(e, 6*nslices(e_ref), 4);
  ko::resize(p, 6*nslices(p_ref));
  for (Int i = 1; i < 6; ++i) {
    ko::resize(es[i], nslices(e_ref), 4);
    ko::deep_copy(es[i], e_ref);
    ko::resize(ps[i], nslices(p_ref));
    ko::deep_copy(ps[i], p_ref);
    transform_planar_mesh(R[i], xlate[i], ps[i]);
  }
  transform_planar_mesh(R[0], xlate[0], ps[0]);
  // Pack (p,e), accounting for equivalent vertices. For the moment, keep the p
  // slot for an equivalent vertex to make node numbering simpler, but make the
  // value bogus so we know if there's a problem in the numbering.
  const Real unused = -2;
  ko::deep_copy(p, unused);
  Int p_base = 0, e_base = 0;
  { // -y face
    const Vec3s::HostMirror& fp = ps[0];
    Idxs::HostMirror& fe = es[0];
    for (Int j = 0; j < nslices(fp); ++j)
      for (Int k = 0; k < 3; ++k) p(j,k) = fp(j,k);
    for (Int j = 0; j < nslices(fe); ++j)
      for (Int k = 0; k < 4; ++k) e(j,k) = fe(j,k);
    p_base += nslices(p_ref);
    e_base += nslices(e_ref);
  }
  for (Int fi = 1; fi <= 2; ++fi) { // +x, +y faces
    const Vec3s::HostMirror& fp = ps[fi];
    Idxs::HostMirror& fe = es[fi];
    for (Int j = 0; j < nslices(fp); ++j) {
      if (j % (n+1) == 0) continue; // equiv vertex
      for (Int k = 0; k < 3; ++k) p(p_base+j,k) = fp(j,k);
    }
    for (Int j = 0; j < nslices(fe); ++j) {
      for (Int k = 0; k < 4; ++k) fe(j,k) += p_base;
      // Left 2 vertices of left elem on face fi equiv to right 2 vertices of
      // right elem on face fi-1. Write to the face, then copy to e, so that
      // other faces can use these updated data.
      if (j % n == 0) {
        fe(j,0) = es[fi-1](j+n-1,1);
        fe(j,3) = es[fi-1](j+n-1,2);
      }
      for (Int k = 0; k < 4; ++k) e(e_base+j,k) = fe(j,k);
    }
    p_base += nslices(p_ref);
    e_base += nslices(e_ref);
  }
  { // -x face
    const Vec3s::HostMirror& fp = ps[3];
    Idxs::HostMirror& fe = es[3];
    for (Int j = 0; j < nslices(fp); ++j) {
      if (j % (n+1) == 0 || (j+1) % (n+1) == 0) continue;
      for (Int k = 0; k < 3; ++k) p(p_base+j,k) = fp(j,k);
    }
    for (Int j = 0; j < nslices(fe); ++j) {
      for (Int k = 0; k < 4; ++k) fe(j,k) += p_base;
      if (j % n == 0) {
        fe(j,0) = es[2](j+n-1,1);
        fe(j,3) = es[2](j+n-1,2);
      } else if ((j+1) % n == 0) {
        fe(j,1) = es[0]((j+1)-n,0);
        fe(j,2) = es[0]((j+1)-n,3);
      }
      for (Int k = 0; k < 4; ++k) e(e_base+j,k) = fe(j,k);
    }
    p_base += nslices(p_ref);
    e_base += nslices(e_ref);
  }
  { // +z face
    const Vec3s::HostMirror& fp = ps[4];
    Idxs::HostMirror& fe = es[4];
    for (Int j = n+1; j < nslices(fp) - (n+1); ++j) {
      if (j % (n+1) == 0 || (j+1) % (n+1) == 0) continue;
      for (Int k = 0; k < 3; ++k) p(p_base+j,k) = fp(j,k);
    }
    for (Int j = 0; j < nslices(fe); ++j)
      for (Int k = 0; k < 4; ++k) fe(j,k) += p_base;
    for (Int j = 0; j < n; ++j) { // -y
      fe(j,0) = es[0](n*(n-1)+j,3);
      fe(j,1) = es[0](n*(n-1)+j,2);
    }
    for (Int j = 0; j < n; ++j) { // +y
      fe(n*(n-1)+j,2) = es[2](n*n-1-j,3);
      fe(n*(n-1)+j,3) = es[2](n*n-1-j,2);
    }
    for (Int j = 0, i3 = 0; j < nslices(fe); j += n, ++i3) { // -x
      fe(j,0) = es[3](n*n-1-i3,2);
      fe(j,3) = es[3](n*n-1-i3,3);
    }
    for (Int j = n-1, i1 = 0; j < nslices(fe); j += n, ++i1) { // +x
      fe(j,1) = es[1](n*(n-1)+i1,3);
      fe(j,2) = es[1](n*(n-1)+i1,2);
    }
    for (Int j = 0; j < nslices(fe); ++j)
      for (Int k = 0; k < 4; ++k) e(e_base+j,k) = fe(j,k);
    p_base += nslices(p_ref);
    e_base += nslices(e_ref);
  }
  { // -z face
    const Vec3s::HostMirror& fp = ps[5];
    Idxs::HostMirror& fe = es[5];
    for (Int j = n+1; j < nslices(fp) - (n+1); ++j) {
      if (j % (n+1) == 0 || (j+1) % (n+1) == 0) continue;
      for (Int k = 0; k < 3; ++k) p(p_base+j,k) = fp(j,k);
    }
    for (Int j = 0; j < nslices(fe); ++j)
      for (Int k = 0; k < 4; ++k) fe(j,k) += p_base;
    for (Int j = 0; j < n; ++j) { // -y
      fe(j,0) = es[0](n-1-j,1);
      fe(j,1) = es[0](n-1-j,0);
    }
    for (Int j = 0; j < n; ++j) { // +y
      fe(n*(n-1)+j,2) = es[2](j,1);
      fe(n*(n-1)+j,3) = es[2](j,0);
    }
    for (Int j = 0, i3 = 0; j < nslices(fe); j += n, ++i3) { // -x
      fe(j,0) = es[1](i3,0);
      fe(j,3) = es[1](i3,1);
    }
    for (Int j = n-1, i1 = 0; j < nslices(fe); j += n, ++i1) { // +x
      fe(j,1) = es[3](n-1-i1,1);
      fe(j,2) = es[3](n-1-i1,0);
    }
    for (Int j = 0; j < nslices(fe); ++j)
      for (Int k = 0; k < 4; ++k) e(e_base+j,k) = fe(j,k);
  }
  // Now go back and remove the unused vertices and adjust the numbering.
  remove_unused_vertices(p, e, unused);
  // Project to the unit sphere.
  for (Int i = 0; i < nslices(p); ++i)
    SphereGeometry::normalize(slice(p, i));
}

void calc_elem_ctr (const Vec3s::HostMirror& p, const Idxs::HostMirror& e,
                    const Int ei, Real ctr[3]) {
  for (Int j = 0; j < 3; ++j) ctr[j] = 0;
  Int n = 0;
  for (Int i = 0; i < szslice(e); ++i) {
    if (e(ei,i) < 0) break;
    for (Int j = 0; j < 3; ++j) ctr[j] += p(e(ei,i),j);
    ++n;
  }
  for (Int j = 0; j < 3; ++j) ctr[j] /= n;
}

// Return 0 if all elements' subtri normals point outward relative to the
// sphere.
Int check_elem_normal_against_sphere (const Vec3s::HostMirror& p,
                                      const Idxs::HostMirror& e) {
  Int nerr = 0;
  for (Int ei = 0; ei < nslices(e); ++ei) { // for each element
    Real sphere[3]; // ray through elem ctr
    calc_elem_ctr(p, e, ei, sphere);
    for (Int ti = 0; ti < szslice(e) - 2; ++ti) { // for each tri
      if (e(ei,ti+2) < 0) break;
      Real tri_normal[3]; {
        Real v[2][3];
        for (Int j = 0; j < 2; ++j) {
          SphereGeometry::copy(v[j], slice(p, e(ei,ti+j+1)));
          SphereGeometry::axpy(-1, slice(p, e(ei,0)), v[j]);
        }
        SphereGeometry::cross(v[0], v[1], tri_normal);
      }
      if (SphereGeometry::dot(tri_normal, sphere) <= 0)
        ++nerr;
    }
  }
  return nerr;
}

//> Unit test code.

struct Input {
  Int testno;
  Int n;
  Real angle, xlate, ylate;
  bool write_matlab, geo_sphere;

  Input(Int argc, char** argv);
  void print(std::ostream& os) const;
};

static void project_onto_sphere (Vec3s::HostMirror& p) {
  for (Int ip = 0; ip < nslices(p); ++ip) {
    p(ip,2) = 1;
    SphereGeometry::normalize(slice(p, ip));
  }
}

static void
perturb_mesh (Vec3s::HostMirror& p, const Real angle, const Real xlate,
              const Real ylate) {
  const Real cr = std::cos(angle), sr = std::sin(angle);
  for (Int ip = 0; ip < nslices(p); ++ip) {
    const Real x = p(ip,0), y = p(ip,1);
    p(ip,0) =  cr*x - sr*y + xlate;
    p(ip,1) = -sr*x + cr*y + ylate;
  }  
}

static void
rotate_mesh (Vec3s::HostMirror& p, const Real axis[3], const Real angle) {
  Real R[9];
  form_rotation(axis, angle, R);
  for (Int i = 0; i < nslices(p); ++i)
    rotate(R, slice(p,i));
}

static void fill_quad (const ConstVec3s::HostMirror& p,
                       Vec3s::HostMirror& poly) {
  const Int n = static_cast<int>(std::sqrt(nslices(p) - 1));
  copy(slice(poly, 0), slice(p, 0), 3);
  copy(slice(poly, 1), slice(p, n), 3);
  copy(slice(poly, 2), slice(p, nslices(p) - 1), 3);
  copy(slice(poly, 3), slice(p, nslices(p) - 1 - n), 3);
}

// Area of the outline of (p,e) clipped against the outline of (cp,ce).
template <typename Geo>
static Real calc_true_area (
  const ConstVec3s::HostMirror& cp, const ConstIdxs::HostMirror& ce,
  const ConstVec3s::HostMirror& p, const ConstIdxs::HostMirror& e,
  const bool wm)
{
  Vec3s::HostMirror clip_poly("clip_poly", 4), poly("poly", 4),
    nml("nml", 4);
  fill_quad(cp, clip_poly);
  fill_quad(p, poly);
  for (Int i = 0; i < 4; ++i)
    Geo::edge_normal(slice(clip_poly, i), slice(clip_poly, (i+1) % 4),
                     slice(nml, i));
  Vec3s::HostMirror vo("vo", test::max_nvert);
  Int no;
  {
    Vec3s::HostMirror wrk("wrk", test::max_nvert);
    sh::clip_against_poly<Geo>(clip_poly, nml, poly, 4, vo, no, wrk);
  }
  if (wm) {
    write_matlab("clip_poly", clip_poly);
    write_matlab("poly", poly);
    write_matlab("intersection",
                 ko::subview(vo, std::pair<Int,Int>(0, no), ko::ALL()));
  }
  return Geo::calc_area_formula(vo, no);
}

template <typename Geo> void finalize_mesh (Vec3s::HostMirror& p) {}
template <> void finalize_mesh<SphereGeometry> (Vec3s::HostMirror& p) {
  project_onto_sphere(p);
}

template <typename Geo>
static Int
test_area (const Int n, const Real angle, const Real xlate, const Real ylate,
           const bool wm) {
  Vec3s::HostMirror cp;
  Idxs::HostMirror ce;
  make_planar_mesh(cp, ce, n);

  Vec3s::HostMirror p; resize_and_copy(p, cp);
  Idxs::HostMirror e; resize_and_copy(e, ce);
  perturb_mesh(p, angle, xlate, ylate);

  finalize_mesh<Geo>(cp);
  finalize_mesh<Geo>(p);

  const Real ta = calc_true_area<Geo>(cp, ce, p, e, wm);
  const Real a = test::test_area_ot<Geo>(cp, ce, p, e);

  const Real re = std::abs(a - ta)/ta;
  fprintf(stderr, "true area %1.4e mesh area %1.4e relerr %1.4e\n", ta, a, re);
  if (wm) {
    write_matlab("cm", cp, ce);
    write_matlab("m", p, e);
  }
  return re < 1e-8 ? 0 : 1;
}

static Int test_cube (const Input& in) {
  Vec3s::HostMirror cp;
  Idxs::HostMirror ce;
  make_cubesphere_mesh(cp, ce, in.n);
  Vec3s::HostMirror p; resize_and_copy(p, cp);
  Idxs::HostMirror e; resize_and_copy(e, ce);
  Int nerr = 0;
  {
    const Int ne = check_elem_normal_against_sphere(cp, ce);
    if (ne) std::cerr << "FAIL: check_elem_normal_against_sphere\n";
    nerr += ne;
  }
  { // Make a copy, perturb it, and compute the area of the sphere from the
    // overlap mesh.
    Real axis[] = {0.1, -0.3, 0.2};
    rotate_mesh(p, axis, in.angle);
    const Real
      a = test::test_area_ot<SphereGeometry>(cp, ce, p, e),
      ta = 4*M_PI,
      re = std::abs(a - ta)/ta;
    fprintf(stderr, "true area %1.4e mesh area %1.4e relerr %1.4e\n",
            ta, a, re);
    nerr += re < 1e-8 ? 0 : 1;
  }
  // Test ref square <-> spherical quad transformations.
  nerr += sqr::test::test_sphere_to_ref(p, e);
  if (in.write_matlab) {
    write_matlab("cm", cp, ce);
    write_matlab("m", p, e);
  }
  return nerr;
}

template <typename Geo>
Int run (const Input& in) {
  switch (in.testno) {
  case 0:
    return test_area<Geo>(in.n, in.angle, in.xlate, in.ylate, in.write_matlab);
  case 1:
    return test_cube(in);
  default:
    return 1;
  }
}

inline bool
eq (const std::string& a, const char* const b1, const char* const b2 = 0) {
  return (a == std::string(b1) || (b2 && a == std::string(b2)) ||
          a == std::string("-") + std::string(b1));
}

Input::Input (Int argc, char** argv)
  : testno(0), n(25), angle(M_PI*1e-1), xlate(1e-1), ylate(1e-1),
    write_matlab(false), geo_sphere(true)
{
  for (Int i = 1; i < argc; ++i) {
    const std::string& token = argv[i];
    if (eq(token, "--testno")) testno = atoi(argv[++i]);
    else if (eq(token, "-n")) n = atoi(argv[++i]);
    else if (eq(token, "-m", "--write-matlab")) write_matlab = true;
    else if (eq(token, "--plane")) geo_sphere = false;
    else if (eq(token, "--xlate")) xlate = atof(argv[++i]);
    else if (eq(token, "--ylate")) ylate = atof(argv[++i]);
    else if (eq(token, "--angle")) angle = atof(argv[++i]);
  }

  print(std::cout);
}

void Input::print (std::ostream& os) const {
  os << "testno " << testno << "\n"
     << "n (-n): " << n << "\n"
     << "write matlab (-m): " << write_matlab << "\n"
     << "planar geometry (--plane): " << ! geo_sphere << "\n"
     << "angle (--angle): " << angle << "\n"
     << "xlate (--xlate): " << xlate << "\n"
     << "ylate (--ylate): " << ylate << "\n";
}

int main (int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Input in(argc, argv);
    Int nerr = 0;
    if (in.geo_sphere)
      nerr += run<SphereGeometry>(in);
    else {
#ifdef INSTANTIATE_PLANE
      nerr += run<PlaneGeometry>(in);
#else
      Kokkos::abort("PlaneGeometry not instantiated.");
#endif
    }
    std::cerr << (nerr ? "FAIL" : "PASS") << "ED\n";
  }
  Kokkos::finalize_all();
}
