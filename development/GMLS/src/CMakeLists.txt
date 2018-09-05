bob_config_header("${CMAKE_CURRENT_BINARY_DIR}/GMLS_Config.h")
set(GMLS_HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Config.h"
  GMLS_boost.hpp
  GMLS_gsl.hpp
  GMLS_Kokkos.hpp
  GMLS_LinearAlgebra_Declarations.hpp
  GMLS_LinearAlgebra_Definitions.hpp
  GMLS_particle_boost.hpp
  GMLS_particle_gsl.hpp
  )


set(GMLS_SOURCES
  GMLS_boost.cpp
  GMLS_gsl.cpp  
  GMLS_Kokkos.cpp
  GMLS_particle_gsl.cpp
  GMLS_particle_boost.cpp
  )

# tests
#function(osh_add_exe EXE_NAME)
#  add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
#  target_link_libraries(${EXE_NAME} PRIVATE omega_h)
#endfunction(osh_add_exe)
#
#macro(osh_add_util EXE_NAME)
#  osh_add_exe(${EXE_NAME})
#  bob_export_target(${EXE_NAME})
#endmacro(osh_add_util)
#
#osh_add_util(GMLS_PointValuesTest_boost)
#osh_add_util(GMLS_PointValuesTest_Kokkos)
#osh_add_util(GMLS_PointValuesTest_Kokkos_Boost_Comparison)

add_library(gmls ${GMLS_SOURCES})
bob_library_includes(gmls)

bob_link_dependency(gmls PUBLIC KokkosCore)
#if (Compadre_USE_KokkosCore)
#  target_include_directories(gmls PUBLIC "${KokkosCore_INCLUDE_DIRS}")
#endif()
if (Compadre_USE_Boost)
  target_include_directories(gmls PUBLIC "${Boost_INCLUDE_DIR}")
endif()



bob_export_target(gmls)


install(FILES ${GMLS_HEADERS} DESTINATION include)

bob_end_subdir()
