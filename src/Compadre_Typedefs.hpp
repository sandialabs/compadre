#ifndef _COMPADRE_TYPEDEFS_HPP_
#define _COMPADRE_TYPEDEFS_HPP_

#include "Compadre_Config.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <type_traits>
#include <vector>
#include <sstream>
#include <cstddef>

/*!
 
  Data types in Compadre Toolkit:

    - Intention is to do local work, i.e. on a single node, so the default ordinal is local_index_type
    - When doing pointer arithmetic, it is possible to overflow local_index_type, so use global_index_type

*/

// Indices and data types
typedef double      scalar_type;
typedef int         local_index_type;
typedef std::size_t global_index_type;

// helper function when doing pointer arithmetic
#define TO_GLOBAL(variable) ((global_index_type)variable)

// KOKKOS TYPEDEFS

// execution spaces
typedef Kokkos::DefaultHostExecutionSpace host_execution_space;
#ifdef COMPADRE_USE_CUDA
  typedef Kokkos::DefaultExecutionSpace device_execution_space;
#else
  typedef Kokkos::DefaultHostExecutionSpace device_execution_space;
#endif

// team policies
typedef typename Kokkos::TeamPolicy<device_execution_space> team_policy;
typedef typename team_policy::member_type  member_type;

typedef typename Kokkos::TeamPolicy<host_execution_space> host_team_policy;
typedef typename host_team_policy::member_type  host_member_type;

// layout types
typedef Kokkos::LayoutRight layout_right;
typedef Kokkos::LayoutLeft layout_left;

// unmanaged data wrappers
typedef Kokkos::View<double**, layout_right, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_matrix_right_type;
typedef Kokkos::View<double**, layout_left, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_matrix_left_type;

typedef Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_vector_type;
typedef Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > scratch_local_index_type;

// random number generator
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
