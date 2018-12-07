#ifndef _GMLS_TYPEDEFS_HPP_
#define _GMLS_TYPEDEFS_HPP_

#include "GMLS_Config.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <assert.h>
#include <type_traits>
#include <vector>

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

#endif
