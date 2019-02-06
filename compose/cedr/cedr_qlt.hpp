// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_CEDR_QLT_HPP
#define INCLUDE_CEDR_QLT_HPP

#include <mpi.h>

#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <list>

#include "cedr_cdr.hpp"
#include "cedr_util.hpp"

namespace cedr {
// QLT: Quasi-local tree-based non-iterative tracer density reconstructor for
//      mass conservation, shape preservation, and tracer consistency.
namespace qlt {
using cedr::mpi::Parallel;

namespace impl {
struct NodeSets {
  typedef std::shared_ptr<const NodeSets> ConstPtr;
  
  enum : int { mpitag = 42 };

  // A node in the tree that is relevant to this rank.
  struct Node {
    // Rank of the node. If the node is in a level, then its rank is my rank. If
    // it's not in a level, then it is a comm partner of a node on this rank.
    Int rank;
    // Globally unique identifier; cellidx if leaf node, ie, if nkids == 0.
    Int id;
    // This node's parent, a comm partner, if such a partner is required.
    Int parent;
    // This node's kids, comm partners, if such partners are required. Parent
    // and kid nodes are pruned relative to the full tree over the mesh to
    // contain just the nodes that matter to this rank.
    Int nkids;
    Int kids[2];
    // Offset factor into bulk data. An offset is a unit; actual buffer sizes
    // are multiples of this unit.
    Int offset;

    Node () : rank(-1), id(-1), parent(-1), nkids(0), offset(-1) {}
  };

  // A level in the level schedule that is constructed to orchestrate
  // communication. A node in a level depends only on nodes in lower-numbered
  // levels (l2r) or higher-numbered (r2l).
  //
  // The communication patterns are as follows:
  //   > l2r
  //   MPI rcv into kids
  //   sum into node
  //   MPI send from node
  //   > r2l
  //   MPI rcv into node
  //   solve QP for kids
  //   MPI send from kids
  struct Level {
    struct MPIMetaData {
      Int rank;   // Rank of comm partner.
      Int offset; // Offset to start of buffer for this comm.
      Int size;   // Size of this buffer in units of offsets.
    };
    
    // The nodes in the level.
    std::vector<Int> nodes;
    // MPI information for this level.
    std::vector<MPIMetaData> me, kids;
    mutable std::vector<mpi::Request> me_send_req, me_recv_req, kids_req;
  };
  
  // Levels. nodes[0] is level 0, the leaf level.
  std::vector<Level> levels;
  // Number of data slots this rank needs. Each node owned by this rank, plus
  // kids on other ranks, have an associated slot.
  Int nslots;
  
  // Allocate a node. The list node_mem_ is the mechanism for memory ownership;
  // node_mem_ isn't used for anything other than owning nodes.
  Int alloc () {
    const Int idx = node_mem_.size();
    node_mem_.push_back(Node());
    return idx;
  }

  Int nnode () const { return node_mem_.size(); }

  Node* node_h (const Int& idx) {
    cedr_assert(idx >= 0 && idx < static_cast<Int>(node_mem_.size()));
    return &node_mem_[idx];
  }
  const Node* node_h (const Int& idx) const {
    return const_cast<NodeSets*>(this)->node_h(idx);
  }

  void print(std::ostream& os) const;
  
private:
  std::vector<Node> node_mem_;
};

template <typename ExeSpace>
struct NodeSetsDeviceData {
  typedef typename cedr::impl::DeviceType<ExeSpace>::type Device;
  typedef Kokkos::View<Int*, Device> IntList;
  typedef Kokkos::View<NodeSets::Node*, Device> NodeList;

  NodeList node;
  // lvl(lvlptr(l):lvlptr(l+1)-1) is the list of node indices into node for
  // level l.
  IntList lvl, lvlptr;
};

typedef impl::NodeSetsDeviceData<Kokkos::DefaultHostExecutionSpace> NodeSetsHostData;
} // namespace impl

namespace tree {
// The caller builds a tree of these nodes to pass to QLT.
struct Node {
  typedef std::shared_ptr<Node> Ptr;
  const Node* parent; // (Can't be a shared_ptr: would be a circular dependency.)
  Int rank;           // Owning rank.
  Long cellidx;       // If a leaf, the cell to which this node corresponds.
  Int nkids;          // 0 at leaf, 1 or 2 otherwise.
  Node::Ptr kids[2];
  Int reserved;       // For internal use.
  Node () : parent(nullptr), rank(-1), cellidx(-1), nkids(0), reserved(-1) {}
};

// Utility to make a tree over a 1D mesh. For testing, it can be useful to
// create an imbalanced tree.
Node::Ptr make_tree_over_1d_mesh(const Parallel::Ptr& p, const Int& ncells,
                                 const bool imbalanced = false);
} // namespace tree

template <typename ExeSpace = Kokkos::DefaultExecutionSpace>
class QLT : public cedr::CDR {
public:
  typedef typename cedr::impl::DeviceType<ExeSpace>::type Device;
  typedef QLT<ExeSpace> Me;
  typedef std::shared_ptr<Me> Ptr;
  
  // Set up QLT topology and communication data structures based on a tree. Both
  // ncells and tree refer to the global mesh, not just this processor's
  // part. The tree must be identical across ranks.
  QLT(const Parallel::Ptr& p, const Int& ncells, const tree::Node::Ptr& tree);

  void print(std::ostream& os) const override;

  // Number of cells owned by this rank.
  Int nlclcells() const;

  // Cells owned by this rank, in order of local numbering. Thus,
  // gci2lci(gcis[i]) == i. Ideally, the caller never actually calls gci2lci(),
  // and instead uses the information from get_owned_glblcells to determine
  // local cell indices.
  void get_owned_glblcells(std::vector<Long>& gcis) const;

  // For global cell index cellidx, i.e., the globally unique ordinal associated
  // with a cell in the caller's tree, return this rank's local index for
  // it. This is not an efficient operation.
  Int gci2lci(const Int& gci) const;

  void declare_tracer(int problem_type, const Int& rhomidx) override;

  void end_tracer_declarations() override;

  int get_problem_type(const Int& tracer_idx) const override;

  Int get_num_tracers() const override;

  // lclcellidx is gci2lci(cellidx).
  KOKKOS_INLINE_FUNCTION
  void set_rhom(const Int& lclcellidx, const Int& rhomidx, const Real& rhom) const override;

  // lclcellidx is gci2lci(cellidx).
  KOKKOS_INLINE_FUNCTION
  void set_Qm(const Int& lclcellidx, const Int& tracer_idx,
              const Real& Qm, const Real& Qm_min, const Real& Qm_max,
              const Real Qm_prev = std::numeric_limits<Real>::infinity()) const override;

  void run() override;

  KOKKOS_INLINE_FUNCTION
  Real get_Qm(const Int& lclcellidx, const Int& tracer_idx) const override;

protected:
  typedef Kokkos::View<Int*, Device> IntList;
  typedef cedr::impl::Const<IntList> ConstIntList;
  typedef cedr::impl::ConstUnmanaged<IntList> ConstUnmanagedIntList;

  static void init(const std::string& name, IntList& d,
                   typename IntList::HostMirror& h, size_t n);

  struct MetaDataBuilder {
    typedef std::shared_ptr<MetaDataBuilder> Ptr;
    std::vector<int> trcr2prob;
  };

PROTECTED_CUDA:
  struct MetaData {
    enum : Int { nprobtypes = 4 };

    template <typename IntListT>
    struct Arrays {
      // trcr2prob(i) is the ProblemType of tracer i.
      IntListT trcr2prob;
      // bidx2trcr(prob2trcrptr(i) : prob2trcrptr(i+1)-1) is the list of
      // tracers having ProblemType index i. bidx2trcr is the permutation
      // from the user's tracer index to the bulk data's ordering (bidx).
      Int prob2trcrptr[nprobtypes+1];
      IntListT bidx2trcr;
      // Inverse of bidx2trcr.
      IntListT trcr2bidx;
      // Points to the start of l2r bulk data for each problem type, within a
      // slot.
      Int prob2bl2r[nprobtypes + 1];
      // Point to the start of l2r bulk data for each tracer, within a slot.
      IntListT trcr2bl2r;
      // Same for r2l bulk data.
      Int prob2br2l[nprobtypes + 1];
      IntListT trcr2br2l;
    };

    KOKKOS_INLINE_FUNCTION static int get_problem_type(const int& idx);
    
    // icpc doesn't let us use problem_type_ here, even though it's constexpr.
    static int get_problem_type_idx(const int& mask);

    KOKKOS_INLINE_FUNCTION static int get_problem_type_l2r_bulk_size(const int& mask);

    static int get_problem_type_r2l_bulk_size(const int& mask);

    struct CPT {
      // We could make the l2r buffer smaller by one entry, Qm. However, the
      // l2r comm is more efficient if it's done with one buffer. Similarly,
      // we separate the r2l data into a separate buffer for packing and MPI
      // efficiency.
      //   There are 7 possible problems.
      //   The only problem not supported is conservation alone. It makes very
      // little sense to use QLT for conservation alone.
      //   The remaining 6 fall into 4 categories of details. These 4 categories
      // are tracked by QLT; which of the original 6 problems being solved is
      // not important.
      enum {
        // l2r: rhom, (Qm_min, Qm, Qm_max)*; l2r, r2l: Qm*
        s  = ProblemType::shapepreserve,
        st = ProblemType::shapepreserve | ProblemType::consistent,
        // l2r: rhom, (Qm_min, Qm, Qm_max, Qm_prev)*; l2r, r2l: Qm*
        cs  = ProblemType::conserve | s,
        cst = ProblemType::conserve | st,
        // l2r: rhom, (q_min, Qm, q_max)*; l2r, r2l: Qm*
        t = ProblemType::consistent,
        // l2r: rhom, (q_min, Qm, q_max, Qm_prev)*; l2r, r2l: Qm*
        ct = ProblemType::conserve | t
      };
    };

    Arrays<typename ConstUnmanagedIntList::HostMirror> a_h;
    Arrays<ConstUnmanagedIntList> a_d;

    void init(const MetaDataBuilder& mdb);

  private:
    Arrays<typename IntList::HostMirror> a_h_;
    Arrays<IntList> a_d_;
  };

  struct BulkData {
    typedef Kokkos::View<Real*, Device> RealList;
    typedef cedr::impl::Unmanaged<RealList> UnmanagedRealList;

    UnmanagedRealList l2r_data, r2l_data;

    void init(const MetaData& md, const Int& nslots);

  private:
    RealList l2r_data_, r2l_data_;
  };

protected:
  void init(const Parallel::Ptr& p, const Int& ncells, const tree::Node::Ptr& tree);

  void init_ordinals();

  /// Pointer data for initialization and host computation.
  Parallel::Ptr p_;
  // Tree and communication topology.
  std::shared_ptr<const impl::NodeSets> ns_;
  // Data extracted from ns_ for use in run() on device.
  std::shared_ptr<impl::NodeSetsDeviceData<ExeSpace> > nsdd_;
  std::shared_ptr<impl::NodeSetsHostData> nshd_;
  // Globally unique cellidx -> rank-local index.
  typedef std::map<Int,Int> Gci2LciMap;
  std::shared_ptr<Gci2LciMap> gci2lci_;
  // Temporary to collect caller's tracer information prior to calling
  // end_tracer_declarations().
  typename MetaDataBuilder::Ptr mdb_;
  /// View data for host and device computation.
  // Constructed in end_tracer_declarations().
  MetaData md_;
  BulkData bd_;

PRIVATE_CUDA:
  void l2r_recv(const impl::NodeSets::Level& lvl, const Int& l2rndps) const;
  void l2r_combine_kid_data(const Int& lvlidx, const Int& l2rndps) const;
  void l2r_send_to_parents(const impl::NodeSets::Level& lvl, const Int& l2rndps) const;
  void root_compute(const Int& l2rndps, const Int& r2lndps) const;
  void r2l_recv(const impl::NodeSets::Level& lvl, const Int& r2lndps) const;
  void r2l_solve_qp(const Int& lvlidx, const Int& l2rndps, const Int& r2lndps) const;
  void r2l_send_to_kids(const impl::NodeSets::Level& lvl, const Int& r2lndps) const;
};

namespace test {
struct Input {
  bool unittest, perftest, write;
  Int ncells, ntracers, tracer_type, nrepeat;
  bool pseudorandom, verbose;
};

Int run_unit_and_randomized_tests(const Parallel::Ptr& p, const Input& in);

Int test_qlt(const Parallel::Ptr& p, const tree::Node::Ptr& tree, const Int& ncells,
             const Int nrepeat = 1,
             // Diagnostic output for dev and illustration purposes. To be
             // clear, no QLT unit test requires output to be checked; each
             // checks in-memory data and returns a failure count.
             const bool write = false,
             const bool verbose = false);
} // namespace test
} // namespace qlt
} // namespace cedr

// These are the definitions that must be visible in the calling translation
// unit, unless Cuda relocatable device code is enabled.
#include "cedr_qlt_inl.hpp"

#endif
