bob_config_header("${CMAKE_CURRENT_BINARY_DIR}/GMLS_Config.h")
set(GMLS_HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Config.h"
  GMLS.hpp
  GMLS_LinearAlgebra_Declarations.hpp
  GMLS_LinearAlgebra_Definitions.hpp
  )


set(GMLS_SOURCES
  GMLS.cpp
  )





add_library(gmls ${GMLS_SOURCES})
bob_library_includes(gmls)

if (KOKKOS_IN_TRILINOS)
  bob_link_dependency(gmls PUBLIC KokkosCore)
else ()
  target_include_directories(gmls PUBLIC "${KokkosCore_INCLUDE_DIRS}")
  target_link_libraries(gmls PUBLIC "${KokkosCore_STATIC_LIB}")
endif()

if (Compadre_USE_Boost)
  target_include_directories(gmls PUBLIC "${Boost_INCLUDE_DIR}")
endif()



bob_export_target(gmls)


install(FILES ${GMLS_HEADERS} DESTINATION include)

bob_end_subdir()
