#ifndef _COMPADRE_TYPEDEFS_HPP_
#define _COMPADRE_TYPEDEFS_HPP_

#include "Compadre_Config.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <type_traits>
#include <vector>
#include <sstream>

// KOKKOS TYPEDEFS

typedef Kokkos::TeamPolicy<>               team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;

typedef Kokkos::DefaultExecutionSpace::array_layout layout_type;

// reorders indices for layout of device
#ifdef COMPADRE_USE_CUDA
#define ORDER_INDICES(i,j) j,i
#else
#define ORDER_INDICES(i,j) i,j
#endif

typedef Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_matrix_type;
typedef Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_vector_type;
typedef Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_local_index_type;

typedef Kokkos::Random_XorShift64_Pool<> pool_type;
typedef typename pool_type::generator_type generator_type;

//! compadre_assert_release is used for assertions that should always be checked, but generally 
//! are not expensive to verify or are not called frequently. 
# define compadre_assert_release(condition) do {                                \
    if ( ! (condition)) {                                               \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": FAIL:\n" << #condition  \
        << "\n";                                                        \
        throw std::logic_error(_ss_.str());                             \
    }                                                                   \
  } while (0)

//! compadre_kernel_assert_release is similar to compadre_assert_release, but is a call on the device, 
//! namely inside of a function marked KOKKOS_INLINE_FUNCTION
# define compadre_kernel_assert_release(condition) do { \
    if ( ! (condition))                         \
      Kokkos::abort(#condition);                \
  } while (0)

//! compadre_assert_debug is used for assertions that are checked in loops, as these significantly
//! impact performance. When NDEBUG is set, these conditions are not checked.
#ifdef COMPADRE_DEBUG
# define compadre_assert_debug(condition) do {                                \
    if ( ! (condition)) {                                               \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": FAIL:\n" << #condition  \
        << "\n";                                                        \
        throw std::logic_error(_ss_.str());                             \
    }                                                                   \
  } while (0)
# define compadre_kernel_assert_debug(condition) do { \
    if ( ! (condition))                         \
      Kokkos::abort(#condition);                \
  } while (0)
#else
#  define compadre_assert_debug(condition)
#  define compadre_kernel_assert_debug(condition)
#endif
//! compadre_kernel_assert_debug is similar to compadre_assert_debug, but is a call on the device, 
//! namely inside of a function marked KOKKOS_INLINE_FUNCTION

#endif
