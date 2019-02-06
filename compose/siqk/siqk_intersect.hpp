// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_SIQK_INTERSECT_HPP
#define INCLUDE_SIQK_INTERSECT_HPP

#include "siqk_defs.hpp"
#include "siqk_geometry.hpp"
#include "siqk_search.hpp"
#include "siqk_quadrature.hpp"

namespace siqk {

// Sutherland-Hodgmann polygon clipping algorithm. Follow Foley, van Dam,
// Feiner, Hughes Fig 3.49.
namespace sh {
/* A mesh is described by the following arrays:
       p: 3 x #nodes, the array of vertices.
       e: max(#verts) x #elems, the array of element base-0 indices.
       nml: 3 x #edges, the array of edge normals.
       en: max(#verts) x #elems, the array of edge-normal base-0 indices.
     e. e indexes p. e(i,j) == -1 in column j indicates that j:end are not used.
     nml. As a mesh is refined, cancellation error makes an edge normal based
   off of an element's vertices increasingly inaccurate. Roughly, if an edge
   subtends angle phi of the sphere, -log10(phi/(2 pi)) digits are lost in the
   edge normal. Therefore, we compute edge normals offline, since in certain
   meshes, they can be computed by an accurate means. E.g., in a cubed-sphere
   mesh, the whole line of a square face can be used to compute the edge
   normal. Furthermore, there are far fewer unique edge normals than edges.
 */
template <typename ES = ko::DefaultExecutionSpace>
struct Mesh {
  typename InExeSpace<ConstVec3s, ES>::type p, nml;
  typename InExeSpace<ConstIdxs, ES>::type e, en;

  Mesh () {}

  Mesh (const Mesh<ko::HostSpace>& m) {
    typename InExeSpace<Vec3s, ES>::type tp, tnml;
    typename InExeSpace<Idxs, ES>::type te, ten;
    resize_and_copy(tp, m.p); p = tp;
    resize_and_copy(tnml, m.nml); nml = tnml;
    resize_and_copy(te, m.e); e = te;
    resize_and_copy(ten, m.en); en = ten;
  }
};

// Generally not a user routine.
template <typename geo, typename CV3s, typename V3s, typename CV>
KOKKOS_INLINE_FUNCTION
bool clip_against_edge (
  // Input vertex list.
  const CV3s& vi, const Int ni,
  // Output vertex list.
  V3s& vo, Int& no,
  // One point of the clip edge.
  const CV ce1,
  // Clip edge's inward-facing normal.
  const CV cen)
{
  Real intersection[3];
  no = 0;
  auto s = const_slice(vi, ni-1);
  for (Int j = 0; j < ni; ++j) {
    auto p = const_slice(vi,j);
    if (geo::inside(p, ce1, cen)) {
      if (geo::inside(s, ce1, cen)) {
        if ( ! geo::output(p, no, vo)) return false;
      } else {
        geo::intersect(s, p, ce1, cen, intersection);
        if ( ! geo::output(intersection, no, vo)) return false;
        if ( ! geo::output(p, no, vo)) return false;
      }
    } else if (geo::inside(s, ce1, cen)) {
      geo::intersect(s, p, ce1, cen, intersection);
      if ( ! geo::output(intersection, no, vo)) return false;
    }
    s = p;
  }
  return true;
}

// Efficient user routine that uses the mesh data structure.
//todo An optimization would be to have 2 clip_against_edge routines. One would
// handle the special case of the first vertex list being in (p,e) format.
template <typename geo, typename MeshT, typename CV3s, typename V3s>
KOKKOS_INLINE_FUNCTION
bool clip_against_poly (
  // Clip mesh. m.e(:,cp_e) is the element, and m.en(:,cp_e) is the
  // corresponding list of normal indices.
  const MeshT& m, const Int cp_e,
  // A list of vertices describing the polygon to clip. The vertices must be in
  // a convention-determined order, such as CCW. vi(:,1:ni-1) are valid entries.
  const CV3s& vi, const Int ni,
  // On output, vo(:,0:no-1) are vertices of the clipped polygon. no is 0 if
  // there is no intersection.
  V3s& vo, Int& no,
  // Workspace. Both vo and wrk must be large enough to hold all generated
  // vertices. If they are not, false is returned.
  V3s& wrk)
{
  Int nos[] = { 0, 0 };
  V3s* vs[] = { &vo, &wrk };

  const auto e = slice(m.e, cp_e);
  const auto en = slice(m.en, cp_e);

  auto nv = szslice(m.e); // Number of vertices in clip polygon.
  while (e[nv-1] == -1) --nv;

  no = 0;
  if (nv % 2 == 0) {
    // Make sure the final vertex output list is in the caller's buffer.
    swap(vs[0], vs[1]);
    swap(nos[0], nos[1]);
  }

  if ( ! clip_against_edge<geo>(vi, ni, *vs[0], nos[0], const_slice(m.p, e[0]),
                                const_slice(m.nml, en[0])))
    return false;
  if ( ! nos[0]) return true;

  for (Int ie = 1, ielim = nv - 1; ; ++ie) {
    if ( ! clip_against_edge<geo>(*vs[0], nos[0], *vs[1], nos[1],
                                  const_slice(m.p, e[ie]),
                                  const_slice(m.nml, en[ie])))
      return false;
    if ( ! nos[1]) return true;
    if (ie == ielim) break;
    swap(vs[0], vs[1]);
    swap(nos[0], nos[1]);
  }

  no = nos[1];
  return true;
}

// Not used for real stuff; just a convenient version for testing. In this
// version, clip_poly is a list of clip polygon vertices. This is instead of the
// mesh data structure.
template <typename geo, typename CV3s_CP, typename CV3s_CEN, typename CV3s_VI,
          typename V3s>
KOKKOS_INLINE_FUNCTION
bool clip_against_poly (
  // Clip polygon.
  const CV3s_CP& clip_poly,
  // Clip polygon edges' inward-facing normals.
  const CV3s_CEN& clip_edge_normals,
  const CV3s_VI& vi, const Int ni,
  V3s& vo, Int& no,
  V3s& wrk)
{
  Int nos[] = { 0, 0 };
  V3s* vs[] = { &vo, &wrk };

  no = 0;
  if (nslices(clip_poly) % 2 == 0) {
    // Make sure the final vertex output list is in the caller's buffer.
    swap(vs[0], vs[1]);
    swap(nos[0], nos[1]);
  }

  if ( ! clip_against_edge<geo>(vi, ni, *vs[0], nos[0],
                                const_slice(clip_poly, 0),
                                const_slice(clip_edge_normals, 0)))
    return false;
  if ( ! nos[0]) return true;

  for (Int ie = 1, ielim = nslices(clip_poly) - 1; ; ++ie) {
    if ( ! clip_against_edge<geo>(*vs[0], nos[0], *vs[1], nos[1],
                                  const_slice(clip_poly, ie),
                                  const_slice(clip_edge_normals, ie)))
      return false;
    if ( ! nos[1]) return true;
    if (ie == ielim) break;
    swap(vs[0], vs[1]);
    swap(nos[0], nos[1]);
  }

  no = nos[1];
  return true;
}
} // namespace sh

namespace test {
static constexpr Int max_nvert = 20;
static constexpr Int max_hits = 25; // Covers at least a 2-halo.

// In practice, we want to form high-quality normals using information about the
// mesh.
template <typename geo>
void fill_normals (sh::Mesh<ko::HostSpace>& m) {
  // Count number of edges.
  Int ne = 0;
  for (Int ip = 0; ip < nslices(m.e); ++ip)
    for (Int iv = 0; iv < szslice(m.e); ++iv)
      if (m.e(ip,iv) == -1) break; else ++ne;
  // Fill.
  Idxs::HostMirror en("en", nslices(m.e), szslice(m.e));
  ko::deep_copy(en, -1);
  Vec3s::HostMirror nml("nml", ne);
  Int ie = 0;
  for (Int ip = 0; ip < nslices(m.e); ++ip)
    for (Int iv = 0; iv < szslice(m.e); ++iv)
      if (m.e(ip,iv) == -1)
        break;
      else {
        // Somewhat complicated next node index.
        const Int iv_next = (iv+1 == szslice(m.e) ? 0 :
                             (m.e(ip,iv+1) == -1 ? 0 : iv+1));
        geo::edge_normal(slice(m.p, m.e(ip, iv)), slice(m.p, m.e(ip, iv_next)),
                         slice(nml, ie));
        en(ip,iv) = ie;
        ++ie;
      }
  m.en = en;
  m.nml = nml;
}

//todo The current approach is to do redundant clips so that the hits buffer can
// be small and static. Need to think about this.
template <typename geo>
class AreaOTFunctor {
  const TriangleQuadrature quad_;
  const sh::Mesh<>& cm_;
  const ConstVec3s& p_;
  const ConstIdxs& e_;
  const Int k_; // Index into (p,e).
  //todo More efficient method that also works on GPU.
  Int hits_[max_hits];
  Int nh_;
  Real area_;

public:
  KOKKOS_INLINE_FUNCTION
  AreaOTFunctor (const sh::Mesh<>& cm, const ConstVec3s& p, const ConstIdxs& e,
                 const Int& k)
    : cm_(cm), p_(p), e_(e), k_(k), nh_(0), area_(0)
  {}

  KOKKOS_INLINE_FUNCTION void operator() (const Int mesh_elem_idx) {
    // Check whether we've clipped against this polygon before and there was a
    // non-0 intersection.
    for (Int i = 0; i < nh_; ++i)
      if (hits_[i] == mesh_elem_idx)
        return;
    // We have not, so do the intersection.
    Int no = 0;
    {
      // Area of all overlapping regions.
      // In and out vertex lists.
      Real buf[9*max_nvert];
      RawVec3s
        vi(buf, max_nvert),
        vo(buf + 3*max_nvert, max_nvert),
        wrk(buf + 6*max_nvert, max_nvert);
      Int ni;
      ni = 0;
      for (Int i = 0; i < szslice(e_); ++i) {
        if (e_(k_,i) == -1) break;
        copy(slice(vi, i), slice(p_, e_(k_,i)), 3);
        ++ni;
      }
      sh::clip_against_poly<geo>(cm_, mesh_elem_idx, vi, ni, vo, no, wrk);
      if (no) area_ += geo::calc_area(quad_, vo, no);
    }
    if (no) {
      // Non-0 intersection, so record.
      if (nh_ == max_hits) Kokkos::abort("max_hits is too small.");
      hits_[nh_++] = mesh_elem_idx;
    }
  }

  KOKKOS_INLINE_FUNCTION const Real& area () const { return area_; }
};

template <typename geo, typename OctreeT>
class TestAreaOTKernel {
  const sh::Mesh<> cm_;
  const OctreeT ot_;
  mutable ConstVec3s p_;
  mutable ConstIdxs e_;

public:
  typedef Real value_type;

  TestAreaOTKernel (const sh::Mesh<ko::HostSpace>& cm,
                    const ConstVec3s::HostMirror& p_hm,
                    const ConstIdxs::HostMirror& e_hm, const OctreeT& ot)
    : cm_(cm), ot_(ot)
  {
    { Vec3s p; resize_and_copy(p, p_hm); p_ = p; }
    { Idxs e; resize_and_copy(e, e_hm); e_ = e; }
  }
  
  // Clip the k'th polygon in (p,e) against mesh cm.
  KOKKOS_INLINE_FUNCTION void operator() (const Int k, Real& area) const {
    // Clipped element bounding box.
    Real ebb[6];
    OctreeT::calc_bb(p_, slice(e_, k), szslice(e_), ebb);
    // Get list of possible overlaps.
    AreaOTFunctor<geo> f(cm_, p_, e_, k);
    //todo Team threads.
    ot_.apply(ebb, f);
    area += f.area();
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type& dst, volatile value_type const& src) const
  { dst += src; }
};

template <typename geo> Real test_area_ot (
  const ConstVec3s::HostMirror& cp, const ConstIdxs::HostMirror& ce,
  const ConstVec3s::HostMirror& p, const ConstIdxs::HostMirror& e)
{
  typedef Octree<geo, 10> OctreeT;

  // Clip mesh and edge normal calculation. (In practice, we'd like to use
  // higher-quality edge normals.)
  sh::Mesh<ko::HostSpace> cm; cm.p = cp; cm.e = ce;
  fill_normals<geo>(cm);

  Real et[2] = {0};
  auto t = tic();
  // Oct-tree over the clip mesh.
  OctreeT ot(cp, ce);
  et[0] = toc(t);

  Real area = 0;
  TestAreaOTKernel<geo, OctreeT> f(cm, p, e, ot);
  t = tic();
  ko::parallel_reduce(nslices(e), f, area);
  et[1] = toc(t);
  print_times("test_area_ot", et, 2);
  return area;
}
} // namespace test
} // namespace siqk

#endif // INCLUDE_SIQK_INTERSECT_HPP
