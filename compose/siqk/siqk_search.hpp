// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_SIQK_SEARCH_HPP
#define INCLUDE_SIQK_SEARCH_HPP

#include "siqk_defs.hpp"
#include "siqk_geometry.hpp"
#include <vector>

namespace siqk {

// Oct-tree. Might do something else better suited to the sphere later.
template <typename Geo, Int max_depth_ = 10>
class Octree {
public:
  enum { max_depth = max_depth_ };
  typedef Real BoundingBox[6];

  struct Options {
    // Do not go beyond max_depth_ depth, including the root and leaf. With this
    // constraInt, try to go deep enough so that a leaf has no more than
    // max_nelem elements.
    Int max_nelem;
    Options () : max_nelem(8) {}
  };

  // Bounding box for a cluster of points ps (possibly vertices).
  template <typename CV3s, typename BB>
  static void calc_bb (const CV3s& ps, const Int np, BB bb) {
    if (np == 0) return;
    for (Int j = 0; j < 3; ++j)
      bb[j] = bb[j+3] = ps(0,j);
    for (Int i = 1; i < np; ++i) {
      for (Int j = 0; j < 3; ++j) {
        bb[j] = min(bb[j], ps(i,j));
        bb[j+3] = max(bb[j+3], ps(i,j));
      }
    }
    pad_bb(bb);
  }

  template <typename CV3s, typename CIV, typename BB>
  KOKKOS_INLINE_FUNCTION
  static void calc_bb (const CV3s& p, const CIV e, const Int ne, BB ebb) {
    for (Int j = 0; j < 3; ++j)
      ebb[j] = ebb[j+3] = p(e[0], j);
    for (Int i = 1; i < ne; ++i) {
      if (e[i] == -1) break;
      for (Int j = 0; j < 3; ++j) {
        ebb[j] = min(ebb[j], p(e[i], j));
        ebb[j+3] = max(ebb[j+3], p(e[i], j));
      }
    }
    pad_bb(ebb);
  }

  // If a bounding box was constructed from vertices of a spherical polygon,
  // expand it to account for the possible protrusion of the sphere.
  template <typename BB>
  KOKKOS_INLINE_FUNCTION
  static void pad_bb (BB bb) {
    if (std::is_same<Geo, PlaneGeometry>::value) return;
    Real hl = 0.5*std::sqrt(square(bb[3] - bb[0]) + square(bb[4] - bb[1]) +
                            square(bb[5] - bb[2]));
    // Limit the half-length to the circle's radius.
    hl = min(1.0, hl);
    // Max distance from a chord of length 2 hl to the unit circle:
    //     hl = sin theta
    //    pad = 1 - cos theta = 1 - sqrt(1 - sin^2 theta) = 1 - sqrt(1 - hl^2).
    const Real pad = 1 - std::sqrt(1 - square(hl));
    for (Int i = 0; i < 3; ++i) bb[  i] -= pad;
    for (Int i = 0; i < 3; ++i) bb[3+i] += pad;
  }

  template <typename CV3s>
  static void calc_bb (const CV3s& ps, BoundingBox bb) {
    calc_bb(ps, nslices(ps), bb);
  }

  template <typename CV3s, typename CIs, typename V6s>
  static void calc_bb (const CV3s& p, const CIs& e, V6s& ebbs) {
    assert(nslices(ebbs) == nslices(e));
    for (Int k = 0, klim = nslices(e); k < klim; ++k)
      calc_bb(p, slice(e, k), szslice(e), slice(ebbs, k));
  }

  // p is a 3xNp array of points. e is a KxNe array of elements. An entry <0 is
  // ignored. All <0 entries must be at the end of an element's list.
  Octree (const ConstVec3s::HostMirror& p, const ConstIdxs::HostMirror& e,
          const Options& o) {
    init(p, e, o);
  }
  Octree (const ConstVec3s::HostMirror& p, const ConstIdxs::HostMirror& e) {
    Options o;
    init(p, e, o);
  }

  Octree() {}
  void init (const ConstVec3s::HostMirror& p, const ConstIdxs::HostMirror& e) {
    Options o;
    init(p, e, o);
  }

  // Apply f to every element in leaf nodes with which bb overlaps. f must have
  // function
  //     void operator(const Int element).
  template <typename CV, typename Functor>
  KOKKOS_INLINE_FUNCTION
  void apply (const CV bb, Functor& f) const {
    if (nslices(nodes_) == 0) {
      for (Int i = 0; i < offsets_[1]; ++i)
        f(elems_[i]);
      return;
    }
#ifdef SIQK_NONRECURSIVE
    // Non-recursive impl.
    {
      // Stack.
      Real snbb[8*max_depth_];
      Int sni[max_depth_], si[max_depth_];
      Int sp = 0;
      // Args for top-level call.
      copy(snbb, bb_, 8);
      sni[sp] = 0;
      si[sp] = 0;
      while (sp >= 0) {
        // Get stack frame's (nbb, ni, current i) values.
        const Int i = si[sp];
        if (i == 8) {
          --sp;
          continue;
        }
        // Increment stored value of i for next iteration. Current value is
        // stored in 'i' above.
        ++si[sp];
        const Int ni = sni[sp];
        const Real* const nbb = snbb + 8*sp;
        // Can use the next stack frame's bb space for a child bb.
        Real* const child_bb = snbb + 8*(sp+1);
        fill_child_bb(nbb, i, child_bb);
        if ( ! do_bb_overlap(child_bb, bb)) continue;
        Int e = nodes_(ni,i);
        if (e < 0) {
          // Leaf, so apply functor to each element.
          e = std::abs(e + 1);
          for (Int k = offsets_[e]; k < offsets_[e+1]; ++k)
            f(elems_[k]);
        } else if (e > 0) {
          // Recurse.
          ++sp;
          sni[sp] = e;
          si[sp] = 0;
        }
      }
    }
#else
    apply_r(0, bb_, bb, f);
#endif
  }

private:
  /* Each node in the oct-tree contains 8 integers, stored in 'nodes'.

     >0 is an index Into 'nodes', pointing to a child node.

     A <=0 entry in 'nodes' indicates a leaf node. If 0, there are no elements
     in the leaf. If <0, the negative of the entry minus 1 is the index of an
     offset array indexing 'elems'.

     Each segment of 'elems' contains a list of element indices covered by a
     leaf node. Element indices refer to the list of elements the caller
     provides during oct-tree construction.
  */

  // Static data structures holding the completed octree.
  //   nodes(:,i) is a list. The list includes children of node i (>0) and leaf
  // node data (<=0).
  //todo Make these const once ready to do full GPU stuff.
  Nodes nodes_;
  // A leaf node corresponding to -k covers elements
  //     elems[offset[k] : offset[k]-1].
  ko::View<int*> offsets_, elems_;
  // Root node's bounding box.
  BoundingBox bb_;

  // Dynamic data structures for construction phase.
  class IntList {
    Int* const buf_;
    Int i_;
  public:
    IntList (Int* const buf) : buf_(buf), i_(0) {}
    void reset () { i_ = 0; }
    void push (const Int& i) { buf_[i_++] = i; }
    Int* data () { return buf_; }
    Int n () const { return i_; }
    const Int& operator[] (const Int& i) const { return buf_[i]; }
  };

  class DynIntList {
    std::vector<Int> buf_;
  public:
    DynIntList () {}
    void push (const Int& i) { buf_.push_back(i); }
    Int& back () { return buf_.back(); }
    Int& operator[] (const size_t i) {
      if (i >= buf_.size())
        buf_.resize(i+1);
      return buf_[i];
    }
    const Int& operator[] (const size_t i) const { return buf_[i]; }
    Int n () const { return static_cast<Int>(buf_.size()); }
    const Int* data () const { return buf_.data(); }
  };

  // Opposite index slot convention.
  class DynNodes {
    std::vector<Int> buf_;
  public:
    Int n () const { return static_cast<Int>(buf_.size()) >> 3; }
    const Int* data () const { return buf_.data(); }
    Int& operator() (const Int& r, const Int& c) {
      const size_t ec = (c+1) << 3;
      if (ec >= buf_.size())
        buf_.resize(ec);
      return const_cast<Int&>(
        const_cast<const DynNodes*>(this)->operator()(r, c));
    }
    const Int& operator() (const Int& r, const Int& c) const {
      assert(((c << 3) + r) >= 0);
      assert(((c << 3) + r) < (Int) buf_.size());
      return buf_[(c << 3) + r];
    }
  };

  void init (const ConstVec3s::HostMirror& p, const ConstIdxs::HostMirror& e,
             const Options& o) {
    if (nslices(e) == 0) return;
    // Get OT's bounding box.
    calc_bb(p, bb_);
    // Get elements' bounding boxes.
    Vec6s::HostMirror ebbs("ebbs", nslices(e));
    calc_bb(p, e, ebbs);
    // Static element lists for work. Each level has active work space.
    std::vector<Int> buf(max_depth_*nslices(e));
    IntList es(buf.data()), wrk(buf.data() + nslices(e));
    for (Int i = 0, ilim = nslices(e); i < ilim; ++i)
      es.push(i);
    // Dynamic element lists.
    DynIntList offsets, elems;
    offsets[0] = 0;
    // Dynamic node data structure.
    DynNodes nodes;
    // Recurse. We don't care about the return value. If it's 0 and nodes.n() ==
    // 0, we'll detect as much in 'apply'.
    init_r(1, bb_, ebbs, o, es, wrk, offsets, elems, nodes);
    // Build the static data structures.
    if (elems.n() == 0) return;
    init_static_ds(nodes, offsets, elems);
  }

  Int init_r (const Int depth, // Tree's depth at this point, including root.
              const BoundingBox& nbb, // My bounding box.
              const ConstVec6s::HostMirror& ebbs, // All elements' bounding boxes.
              const Options& o, // Options controlling construct of the tree.
              IntList& es, // List of elements in my bounding box.
              IntList& wrk, // Work space to store working element lists.
              DynIntList& offsets, // Offsetss Into elems.
              DynIntList& elems, // Elements belonging to leaf nodes.
              DynNodes& nodes) // Dynamic nodes data structure.
  {
    const Int my_idx = nodes.n(); // My node index.
    // Decide what to do.
    if (es.n() == 0) {
      // I have no elements, so return 0 to indicate I'm a leaf node containing
      // nothing.
      return 0;
    } else if (es.n() <= o.max_nelem || depth == max_depth_) {
      // I'm a leaf node with elements. Store my list of elements and return the
      // storage location.
      const Int os = offsets.back();
      offsets.push(os + es.n());
      for (Int i = 0, n = es.n(); i < n; ++i)
        elems[os + i] = es[i];
      return 1 - offsets.n();
    } else {
      // I'm not a leaf node.
      nodes(0, my_idx) = 0; // Insert myself Into the nodes array.
      for (Int ic = 0; ic < 8; ++ic) {
        BoundingBox child_bb;
        fill_child_bb(nbb, ic, child_bb);
        // Find the elements that are in this child's bb.
        IntList ces(wrk.data());
        for (Int i = 0, n = es.n(); i < n; ++i)
          if (do_bb_overlap(child_bb, slice(ebbs, es[i])))
            ces.push(es[i]);
        // Create some work space.
        IntList cwrk(wrk.data() + ces.n());
        // Recurse.
        const Int child_idx = init_r(depth+1, child_bb, ebbs, o, ces, cwrk,
                                     offsets, elems, nodes);
        nodes(ic, my_idx) = child_idx;
      }
      return my_idx;
    }
  }

  void init_static_ds (const DynNodes nodes, const DynIntList& offsets,
                       const DynIntList& elems) {
    {
      ko::resize(nodes_, nodes.n());
      auto nodes_hm = ko::create_mirror_view(nodes_);
      for (Int i = 0; i < nodes.n(); ++i)
        for (Int j = 0; j < 8; ++j)
          nodes_hm(i,j) = nodes(j,i);
      ko::deep_copy(nodes_, nodes_hm);
    }
    hm_resize_and_copy(offsets_, offsets, offsets.n());
    hm_resize_and_copy(elems_, elems, elems.n());
  }

  // Using parent bb p, fill child bb c, with child_idx in 0:7.
  template <typename CBB, typename BB>
  KOKKOS_INLINE_FUNCTION
  static void fill_child_bb (const CBB& p, const Int& child_idx, BB& c) {
    const Real m[] = { 0.5*(p[0] + p[3]),
                         0.5*(p[1] + p[4]),
                         0.5*(p[2] + p[5]) };
    switch (child_idx) {
    case 0: c[0] = p[0]; c[1] = p[1]; c[2] = p[2]; c[3] = m[0]; c[4] = m[1]; c[5] = m[2]; break;
    case 1: c[0] = m[0]; c[1] = p[1]; c[2] = p[2]; c[3] = p[3]; c[4] = m[1]; c[5] = m[2]; break;
    case 2: c[0] = m[0]; c[1] = m[1]; c[2] = p[2]; c[3] = p[3]; c[4] = p[4]; c[5] = m[2]; break;
    case 3: c[0] = p[0]; c[1] = m[1]; c[2] = p[2]; c[3] = m[0]; c[4] = p[4]; c[5] = m[2]; break;
    case 4: c[0] = p[0]; c[1] = p[1]; c[2] = m[2]; c[3] = m[0]; c[4] = m[1]; c[5] = p[5]; break;
    case 5: c[0] = m[0]; c[1] = p[1]; c[2] = m[2]; c[3] = p[3]; c[4] = m[1]; c[5] = p[5]; break;
    case 6: c[0] = m[0]; c[1] = m[1]; c[2] = m[2]; c[3] = p[3]; c[4] = p[4]; c[5] = p[5]; break;
    case 7: c[0] = p[0]; c[1] = m[1]; c[2] = m[2]; c[3] = m[0]; c[4] = p[4]; c[5] = p[5]; break;
    default:
      // impossible
      error("fill_child_bb: The impossible has happened.");
    }
  }

  // Do bounding boxes a and b overlap?
  template <typename BB>
  KOKKOS_INLINE_FUNCTION
  static bool do_bb_overlap (const BoundingBox a, const BB b) {
    for (Int i = 0; i < 3; ++i)
      if ( ! do_lines_overlap(a[i], a[i+3], b[i], b[i+3]))
        return false;
    return true;
  }

  KOKKOS_INLINE_FUNCTION
  static bool do_lines_overlap (const Real& a1, const Real& a2,
                                const Real& b1, const Real& b2) {
    return ! (a2 < b1 || a1 > b2);
  }

  template <typename CV, typename Functor> KOKKOS_INLINE_FUNCTION
  void apply_r (const Int ni, const BoundingBox& nbb, const CV bb,
                Functor& f) const {
    for (Int i = 0; i < 8; ++i) {
      BoundingBox child_bb;
      fill_child_bb(nbb, i, child_bb);
      if ( ! do_bb_overlap(child_bb, bb)) continue;
      Int e = nodes_(ni,i);
      if (e > 0)
        apply_r(e, child_bb, bb, f);
      else if (e < 0) {
        e = std::abs(e + 1);
        for (Int k = offsets_[e]; k < offsets_[e+1]; ++k)
          f(elems_[k]);
      }
    }
  }
};

} // namespace siqk

#endif // INCLUDE_SIQK_SEARCH_HPP
