bob_config_header("${CMAKE_CURRENT_BINARY_DIR}/GMLS_Config.h")
set(GMLS_HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Config.h"
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
  Compadre_Evaluator.hpp
  GMLS_Typedefs.hpp
  USER_StandardTargetFunctionals.hpp
  USER_ManifoldTargetFunctionals.hpp
  )

set(GMLS_SOURCES
  Compadre_GMLS.cpp
  Compadre_LinearAlgebra.cpp
  )

add_library(gmls ${GMLS_SOURCES})
bob_library_includes(gmls)

if (KOKKOS_IN_TRILINOS)
  bob_link_dependency(gmls PUBLIC KokkosCore)
else ()
  target_include_directories(gmls PUBLIC "${KokkosCore_INCLUDE_DIRS}")
  target_link_libraries(gmls PUBLIC "${KokkosCore_STATIC_LIB}")
endif()

if (Compadre_USE_LAPACK)
  target_link_libraries(gmls PUBLIC "${LAPACK_LIB}")
endif()

if (KokkosCore_HAS_CUDA OR (KOKKOS_INTERNAL_USE_CUDA EQUAL "1"))
  target_include_directories(gmls PUBLIC "${CUDA_INCLUDE_DIRS}")
  target_link_libraries(gmls PUBLIC "${CUDA_CUBLAS_LIBRARIES}")
  target_link_libraries(gmls PUBLIC "${CUDA_CUSOLVER_LIBRARIES}")
  MESSAGE(STATUS "${CUDA_INCLUDE_DIRS}")
endif()

if (Compadre_USE_MPI)
  target_link_libraries(gmls PUBLIC mpi)
endif()

bob_export_target(gmls)

set(GMLS_TPL
    tpl/nanoflann.hpp
)

install(FILES ${GMLS_TPL} DESTINATION include/tpl)

install(FILES ${GMLS_HEADERS} DESTINATION include)


if(Compadre_USE_PYTHON)
  add_subdirectory(swig)
endif()

bob_end_subdir()
