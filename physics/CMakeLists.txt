set(Physics_HEADERS
  Compadre_GMLS_PinnedLaplacian_BoundaryConditions.hpp
  Compadre_GMLS_PinnedLaplacian_Operator.hpp
  Compadre_GMLS_PinnedLaplacian_Sources.hpp
  Compadre_LagrangianShallowWater_BoundaryConditions.hpp
  Compadre_LagrangianShallowWater_Operator.hpp
  Compadre_LagrangianShallowWater_Sources.hpp
  Compadre_LaplaceBeltrami_BoundaryConditions.hpp
  Compadre_LaplaceBeltrami_Operator.hpp
  Compadre_LaplaceBeltrami_Sources.hpp
  Compadre_PinnedGraphLaplacian_BoundaryConditions.hpp
  Compadre_PinnedGraphLaplacian_Operator.hpp
  Compadre_PinnedGraphLaplacian_Sources.hpp
  )


set(Physics_SOURCES
  Compadre_GMLS_PinnedLaplacian_BoundaryConditions.cpp
  Compadre_GMLS_PinnedLaplacian_Operator.cpp
  Compadre_GMLS_PinnedLaplacian_Sources.cpp
  Compadre_LagrangianShallowWater_BoundaryConditions.cpp
  Compadre_LagrangianShallowWater_Operator.cpp
  Compadre_LagrangianShallowWater_Sources.cpp
  Compadre_LaplaceBeltrami_BoundaryConditions.cpp
  Compadre_LaplaceBeltrami_Operator.cpp
  Compadre_LaplaceBeltrami_Sources.cpp
  Compadre_PinnedGraphLaplacian_BoundaryConditions.cpp
  Compadre_PinnedGraphLaplacian_Operator.cpp
  Compadre_PinnedGraphLaplacian_Sources.cpp
  )


add_library(physics ${Physics_SOURCES})
bob_library_includes(physics)

target_link_libraries(physics PRIVATE compadre)

bob_export_target(physics)


install(FILES ${Physics_HEADERS} DESTINATION include)

bob_end_subdir()
