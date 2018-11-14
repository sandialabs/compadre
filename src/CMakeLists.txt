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
  Compadre_Remap.hpp
  )


set(COMPADRE_SOURCES
  Compadre_GMLS.cpp
  Compadre_LinearAlgebra.cpp
  )


add_library(compadre ${COMPADRE_SOURCES})
bob_library_includes(compadre)

if (KOKKOS_IN_TRILINOS)
  bob_link_dependency(compadre PUBLIC KokkosCore)
else ()
  target_include_directories(compadre PUBLIC "${KokkosCore_INCLUDE_DIRS}")
  target_link_libraries(compadre PUBLIC "${KokkosCore_STATIC_LIB}")
endif()

if (Compadre_USE_LAPACK)
  target_link_libraries(compadre PUBLIC "${LAPACK_LIB}")
endif()

if (KokkosCore_HAS_CUDA OR (KOKKOS_INTERNAL_USE_CUDA EQUAL "1"))
  target_include_directories(compadre PUBLIC "${CUDA_INCLUDE_DIRS}")
  target_link_libraries(compadre PUBLIC "${CUDA_CUBLAS_LIBRARIES}")
  target_link_libraries(compadre PUBLIC "${CUDA_CUSOLVER_LIBRARIES}")
  MESSAGE(STATUS "${CUDA_INCLUDE_DIRS}")
endif()

if (Compadre_USE_MPI)
  target_link_libraries(compadre PUBLIC mpi)
endif()

bob_export_target(compadre)

set(COMPADRE_TPL
    tpl/nanoflann.hpp
)

install(FILES ${COMPADRE_TPL} DESTINATION include/tpl)

install(FILES ${COMPADRE_HEADERS} DESTINATION include)

bob_end_subdir()
