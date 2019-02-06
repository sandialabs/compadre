// COMPOSE version 1.0: Copyright 2018 NTESS. This software is released under
// the BSD license; see LICENSE in the top-level directory.

#ifndef INCLUDE_SIQK_DEFS_HPP
#define INCLUDE_SIQK_DEFS_HPP

#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include <Kokkos_Core.hpp>

#ifdef SIQK_TIME
# include <unistd.h>
# include <sys/time.h>
# include <sys/resource.h>
#endif

// Always want this for GPU.
#define SIQK_NONRECURSIVE

#ifdef KOKKOS_HAVE_CUDA
# define KOKKOS_CONSTANT __constant__ __device__
#else
# define KOKKOS_CONSTANT
#endif

namespace siqk {
namespace ko = Kokkos;
#define pr(m) do {                              \
    std::stringstream _ss_;                     \
    _ss_ << m << std::endl;                     \
    std::cerr << _ss_.str();                    \
  } while (0)
#define prc(m) pr(#m << " | " << (m))
#define puf(m)"(" << #m << " " << (m) << ")"
#define pu(m) << " " << puf(m)
template<typename T>
static void prarr (const std::string& name, const T* const v, const size_t n) {
  std::cerr << name << ": ";
  for (size_t i = 0; i < n; ++i) std::cerr << " " << v[i];
  std::cerr << "\n";
}

#define SIQK_THROW_IF(condition, message) do {                          \
    if (condition) {                                                    \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": The condition:\n" << #condition \
        "\nled to the exception\n" << message << "\n";                  \
      throw std::logic_error(_ss_.str());                               \
    }                                                                   \
  } while (0)

#define SIQK_STDERR_IF(condition, message) do { \
  try { SIQK_THROW_IF(condition, message); } \
  catch (const std::logic_error& e) { std::cerr << e.what(); } \
} while (0)

#ifdef SIQK_TIME
static timeval tic () {
  timeval t;
  gettimeofday(&t, 0);
  return t;
}
static double calc_et (const timeval& t1, const timeval& t2) {
  static const double us = 1.0e6;
  return (t2.tv_sec * us + t2.tv_usec - t1.tv_sec * us - t1.tv_usec) / us;
}
static double toc (const timeval& t1) {
  Kokkos::fence();
  timeval t;
  gettimeofday(&t, 0);
  return calc_et(t1, t);
}
static double get_memusage () {
  static const double scale = 1.0 / (1 << 10); // Memory in MB.
  rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss*scale;
}
#else
inline int tic () { return 0; }
inline double toc (const int&) { return 0; }
inline double get_memusage () { return 0; }
#endif
static void print_times (const std::string& name, const double* const parts,
                         const int nparts) {
#ifdef SIQK_TIME
  double total = 0; for (int i = 0; i < nparts; ++i) total += parts[i];
  printf("%20s %1.3e s %7.1f MB", name.c_str(), total, get_memusage());
  for (int i = 0; i < nparts; ++i) printf(" %1.3e s", parts[i]);
  printf("\n");
#endif
}
static void print_times (const std::string& name, const double total) {
#ifdef SIQK_TIME
   printf("%20s %1.3e s %5.1f MB\n", name.c_str(), total, get_memusage());
#endif
}

KOKKOS_INLINE_FUNCTION static void error (const char* const msg)
{ ko::abort(msg); }

KOKKOS_INLINE_FUNCTION static void message (const char* const msg)
{ printf("%s\n", msg); }

typedef int Int;
typedef double Real;

#ifdef KOKKOS_HAVE_CUDA
typedef ko::LayoutLeft Layout;
#else
typedef ko::LayoutRight Layout;
#endif

// SIQK's array types.
typedef ko::View<Real*[3], Layout> Vec3s;
typedef ko::View<const Real*[3], Layout> ConstVec3s;
typedef ko::View<Real*[6], Layout> Vec6s;
typedef ko::View<const Real*[6], Layout> ConstVec6s;
typedef ko::View<Real*[3], ko::LayoutRight, ko::MemoryTraits<ko::Unmanaged> > RawVec3s;
typedef ko::View<const Real*[3], ko::LayoutRight, ko::MemoryTraits<ko::Unmanaged> > RawConstVec3s;
typedef ko::View<Real*, ko::LayoutRight, ko::MemoryTraits<ko::Unmanaged> > RawArray;
typedef ko::View<const Real*, ko::LayoutRight, ko::MemoryTraits<ko::Unmanaged> > RawConstArray;
typedef ko::View<Int**, Layout> Idxs;
typedef ko::View<const Int**, Layout> ConstIdxs;
typedef ko::View<Int*[8], Layout> Nodes;
typedef ko::View<const Int*[8], Layout> ConstNodes;

// Decorator for a View. UnmanagedView<ViewType> gives the same view as
// ViewType, except the memory is unmanaged.
template <typename ViewT>
using UnmanagedView = ko::View<
  typename ViewT::data_type, typename ViewT::array_layout,
  typename ViewT::device_type, ko::MemoryTraits<ko::Unmanaged> >;

// Get the host or device version of the array.
template <typename VT, typename ES> struct InExeSpace {
  typedef VT type;
};
template <typename VT> struct InExeSpace<VT, ko::HostSpace> {
  typedef typename VT::HostMirror type;
};

#ifdef KOKKOS_HAVE_CUDA
// A 1D slice of an array.
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
ko::View<typename VT::value_type*, ko::LayoutStride, typename VT::device_type,
         ko::MemoryTraits<ko::Unmanaged> >
slice (const VT& v, Int i) { return ko::subview(v, i, ko::ALL()); }
// An explicitly const 1D slice of an array.
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
ko::View<typename VT::const_value_type*, ko::LayoutStride, typename VT::device_type,
         ko::MemoryTraits<ko::Unmanaged> >
const_slice (const VT& v, Int i) { return ko::subview(v, i, ko::ALL()); }
#else
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
typename VT::value_type*
slice (const VT& v, Int i) { return v.data() + v.extent(1)*i; }

template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
typename VT::const_value_type*
const_slice (const VT& v, Int i) { return v.data() + v.extent(1)*i; }
#endif

// Number of slices in a 2D array, where each row is a slice.
template <typename A2D> KOKKOS_FORCEINLINE_FUNCTION
Int nslices (const A2D& a) { return static_cast<Int>(a.extent(0)); }

// Number of entries in a 2D array's row.
template <typename A2D> KOKKOS_FORCEINLINE_FUNCTION
Int szslice (const A2D& a) { return static_cast<Int>(a.extent(1)); }

template <typename V, typename CV>
KOKKOS_INLINE_FUNCTION
static void copy (V dst, CV src, const Int n) {
  for (Int i = 0; i < n; ++i) dst[i] = src[i];
}

template <typename DV, typename SV>
void resize_and_copy (DV& d, const SV& s,
                      typename std::enable_if<DV::rank == 1>::type* = 0) {
  ko::resize(d, nslices(s));
  ko::deep_copy(d, s);
}

template <typename DV, typename SV>
void resize_and_copy (
  DV& d, const SV& s,
  typename std::enable_if<DV::rank == 2 && DV::rank_dynamic == 1>::type* = 0)
{
  ko::resize(d, nslices(s));
  ko::deep_copy(d, s);
}

template <typename DV, typename SV>
void resize_and_copy (
  DV& d, const SV& s,
  typename std::enable_if<DV::rank == 2 && DV::rank_dynamic == 2>::type* = 0)
{
  ko::resize(d, nslices(s), szslice(s));
  ko::deep_copy(d, s);
}

template <typename DV, typename SA>
void hm_resize_and_copy (DV& d, const SA& s, const Int n) {
  ko::resize(d, n);
  auto d_hm = ko::create_mirror_view(d);
  for (Int i = 0; i < n; ++i) d_hm[i] = s[i];
  ko::deep_copy(d, d_hm);
}

// GPU-friendly replacements for std::min/max.
template <typename T> KOKKOS_INLINE_FUNCTION
const T& min (const T& a, const T& b) { return a < b ? a : b; }
template <typename T> KOKKOS_INLINE_FUNCTION
const T& max (const T& a, const T& b) { return a > b ? a : b; }
template <typename T> KOKKOS_INLINE_FUNCTION
void swap (T& a, T&b) {
  T tmp = a;
  a = b;
  b = tmp;
}
template <typename T> KOKKOS_INLINE_FUNCTION constexpr T square (const T& x) { return x*x; }

template<typename T> KOKKOS_INLINE_FUNCTION
T sign (const T& a) { return a > 0 ? 1 : (a < 0 ? -1 : 0); }

} // namespace siqk

#endif // INCLUDE_SIQK_DEFS_HPP
