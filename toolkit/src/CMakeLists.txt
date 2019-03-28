bob_config_header("${CMAKE_CURRENT_BINARY_DIR}/Compadre_Config.h")

set(COMPADRE_HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}/Compadre_Config.h"
  Compadre_GMLS.hpp
  Compadre_GMLS_ApplyTargetEvaluations.hpp
  Compadre_GMLS_Basis.hpp
  Compadre_GMLS_Quadrature.hpp
  Compadre_GMLS_Targets.hpp
  Compadre_LinearAlgebra_Declarations.hpp
  Compadre_LinearAlgebra_Definitions.hpp
  Compadre_Misc.hpp
  Compadre_Operators.hpp
  Compadre_PointCloudSearch.hpp
  Compadre_Typedefs.hpp
  Compadre_Evaluator.hpp
  USER_StandardTargetFunctionals.hpp
  USER_ManifoldTargetFunctionals.hpp
  )


set(COMPADRE_SOURCES
  Compadre_GMLS.cpp
  Compadre_LinearAlgebra.cpp
  )


add_library(compadre ${COMPADRE_SOURCES})
bob_library_includes(compadre)

# link to Kokkos
if (KOKKOS_EXISTING_ELSEWHERE) # user points to Kokkos installation
  target_include_directories(compadre PUBLIC "${KokkosCore_INCLUDE_DIRS}")
  target_link_libraries(compadre PUBLIC "${KokkosCore_STATIC_LIB}")
elseif (KOKKOS_IN_TRILINOS) # Kokkos found in Trilinos installation
  bob_link_dependency(compadre PUBLIC KokkosCore)
elseif (KOKKOS_BUILT_FOR_USER) # Kokkos built for user by this project
  # adds a KokkosCore dependency to compadre, but checks whether or not CMake needs to rescan after
  # building KokkosCore
  add_dependencies(compadre RescanKokkosCore)
  target_link_libraries(compadre PUBLIC "${KokkosCore_STATIC_LIB}")
endif()

# link to CUDA
if (Compadre_USE_CUDA AND NOT(KOKKOS_IN_TRILINOS))
  target_include_directories(compadre PUBLIC "${CUDA_INCLUDE_DIRS}")
  target_link_libraries(compadre PUBLIC "${CUDA_CUBLAS_LIBRARIES}")
  target_link_libraries(compadre PUBLIC "${CUDA_CUSOLVER_LIBRARIES}")
  target_link_libraries(compadre PUBLIC "${CUDA_CUDART_LIBRARIES}")
  MESSAGE(STATUS "${CUDA_INCLUDE_DIRS}")
endif()

# link to LAPACK
if (Compadre_USE_LAPACK)
  target_link_libraries(compadre PUBLIC "${LAPACK_LIB}")
endif()

# link to MPI
if (Compadre_USE_MPI)
  if (MPI_CXX_INCLUDE_PATH)
    target_include_directories(compadre PUBLIC "${MPI_CXX_INCLUDE_PATH}")
  endif()
  if (MPI_CXX_LIBRARIES)
    target_link_libraries(compadre PUBLIC "${MPI_CXX_LIBRARIES}")
  endif()
endif()

bob_export_target(compadre)

set(COMPADRE_TPL
    tpl/nanoflann.hpp
)

install(FILES ${COMPADRE_TPL} DESTINATION include/tpl)

install(FILES ${COMPADRE_HEADERS} DESTINATION include)

bob_end_subdir()
