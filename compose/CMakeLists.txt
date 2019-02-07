cmake_minimum_required (VERSION 3.5)

project (compose CXX)
set (CMAKE_CXX_STANDARD 11)

function (prc var)
  message ("${var}: ${${var}}")
endfunction ()

function (list2str list str)
  string (REPLACE ";" " " tmp "${list}")
  set (${str} ${tmp} PARENT_SCOPE)
endfunction ()

set (COMPOSE_DEBUG_MPI FALSE CACHE LOGICAL "If true, insert debugging code into MPI wrappers.")
set (COMPOSE_MIMIC_GPU FALSE CACHE LOGICAL "If true, use non-optimal OpenMP threading to debug GPU-like parallelism.")
set (COMPOSE_QLT_TIME  FALSE CACHE LOGICAL "If true, time QLT operations and display timers at the end.")

if (Kokkos_DIR)
  include (${Kokkos_DIR}/kokkos_generated_settings.cmake)
  if (KOKKOS_IN_TRILINOS)
    set (Kokkos_INCLUDE ${Kokkos_DIR}/../../../include)
  else()
    set (Kokkos_INCLUDE ${Kokkos_DIR}/include)
  endif()
  list2str ("${KOKKOS_CXXFLAGS}" Kokkos_CXX_FLAGS)
  list2str ("${KOKKOS_LD_FLAGS}" Kokkos_LD_FLAGS)
  string (FIND "${KOKKOS_GMAKE_DEVICES}" "Cuda" cuda_str_pos)
  set (CUDA_BUILD FALSE)
  if (${cuda_str_pos} GREATER -1)
    set (CUDA_BUILD TRUE)
  endif ()
else ()
  message (FATAL_ERROR "COMPOSE requires Kokkos_DIR")
endif ()

set (SOURCES
  cedr/cedr_caas.cpp
  cedr/cedr_local.cpp
  cedr/cedr_mpi.cpp
  cedr/cedr_qlt.cpp
  cedr/cedr_test.cpp
  cedr/cedr_test_1d_transport.cpp
  cedr/cedr_test_randomized.cpp
  cedr/cedr_util.cpp)

set (HEADERS
  cedr/cedr.hpp
  cedr/cedr_caas.hpp
  cedr/cedr_caas_inl.hpp
  cedr/cedr_cdr.hpp
  cedr/cedr_kokkos.hpp
  cedr/cedr_local.hpp
  cedr/cedr_local_inl.hpp
  cedr/cedr_mpi.hpp
  cedr/cedr_mpi_inl.hpp
  cedr/cedr_qlt.hpp
  cedr/cedr_qlt_inl.hpp
  cedr/cedr_test.hpp
  cedr/cedr_test_randomized.hpp
  cedr/cedr_util.hpp
  siqk/siqk.hpp
  siqk/siqk_defs.hpp
  siqk/siqk_geometry.hpp
  siqk/siqk_intersect.hpp
  siqk/siqk_quadrature.hpp
  siqk/siqk_search.hpp
  siqk/siqk_sqr.hpp
  share/compose_config.hpp)

if (NOT COMPOSE_TEST_MPIRUN)
  set (COMPOSE_TEST_MPIRUN mpirun)
  set (COMPOSE_TEST_MPIFLAGS)
endif ()
if (NOT COMPOSE_TEST_NRANK)
  set (COMPOSE_TEST_NRANK 8)
endif ()

string (TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_str)
set (DEBUG_BUILD FALSE)
if ("${cmake_build_type_str}" STREQUAL "debug")
  set (DEBUG_BUILD TRUE)
endif ()

if (DEBUG_BUILD)
  message ("Enable COMPOSE_DEBUG_MPI and COMPOSE_MIMIC_GPU because build type is DEBUG.")
  set (COMPOSE_DEBUG_MPI TRUE)
  set (COMPOSE_MIMIC_GPU TRUE)
endif ()

set (COMPOSE_COMPILE_FLAGS "-g ${Kokkos_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
if (CUDA_BUILD)
  set (COMPOSE_COMPILE_FLAGS "${COMPOSE_COMPILE_FLAGS} --expt-relaxed-constexpr")
endif ()
if (KOKKOS_IN_TRILINOS)
  set (COMPOSE_LINK_FLAGS "-L${Kokkos_DIR}/../../../lib ${Kokkos_LD_FLAGS}")
else()
  set (COMPOSE_LINK_FLAGS "-L${Kokkos_DIR}/lib ${Kokkos_LD_FLAGS}")
endif()
set (COMPOSE_INCLUDES ${Kokkos_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/share ${CMAKE_BINARY_DIR}/config)
if (KOKKOS_IN_TRILINOS)
  set (COMPOSE_LIBRARIES "-lkokkoscore")
  set(CMAKE_PREFIX_PATH "${Kokkos_DIR}/../../../lib" ${CMAKE_PREFIX_PATH})
  FIND_LIBRARY(KOKKOSCORE_LIB kokkoscore)
else()
  set (COMPOSE_LIBRARIES ${KOKKOS_LIBS_LIST})
endif()

prc (COMPOSE_COMPILE_FLAGS)
prc (COMPOSE_LINK_FLAGS)
prc (COMPOSE_INCLUDES)
prc (COMPOSE_LIBRARIES)
prc (COMPOSE_DEBUG_MPI)

add_definitions (-DCOMPOSE_CONFIG_IS_CMAKE)
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/compose_config.h.in ${CMAKE_BINARY_DIR}/config/compose_config.h)
set(COMPOSE_CONFIG_FILES
  ${CMAKE_CURRENT_BINARY_DIR}/config/compose_config.h
)
install(FILES ${COMPOSE_CONFIG_FILES} DESTINATION include/compose)

add_library (${PROJECT_NAME} SHARED ${SOURCES})
set_target_properties (${PROJECT_NAME} PROPERTIES
  COMPILE_FLAGS "${COMPOSE_COMPILE_FLAGS}"
  LINK_FLAGS "${COMPOSE_LINK_FLAGS}")
#  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
target_include_directories (${PROJECT_NAME} PUBLIC cedr siqk)
target_include_directories (${PROJECT_NAME} PRIVATE siqk cedr)
target_include_directories (${PROJECT_NAME} PUBLIC ${COMPOSE_INCLUDES})
target_link_libraries (${PROJECT_NAME} PUBLIC ${COMPOSE_LIBRARIES})
if (KOKKOS_IN_TRILINOS)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${KOKKOSCORE_LIB})
endif()


install (TARGETS ${PROJECT_NAME} DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/compose)

enable_testing ()
add_subdirectory(siqk)
add_subdirectory(cedr)
