
# everything requires Tpetra, Zoltan2, TeuchosComm, and TeuchosParameterList
# ProblemT requires linear solvers, i.e. SolverT.cpp
# SolveT requires 
set(CompadreHarness_SOURCES
  Compadre_AnalyticFunctions.cpp     
  Compadre_BoundaryConditionsT.cpp   
  Compadre_CoordsT.cpp               
  Compadre_DOFData.cpp               
  Compadre_DOFManager.cpp            
  Compadre_EuclideanCoordsT.cpp      
  Compadre_FieldManager.cpp          
  Compadre_FieldT.cpp                
  Compadre_FileIO.cpp                
  Compadre_LagCoordsT.cpp            
  Compadre_MultiJumpNeighborhood.cpp         
  Compadre_NeighborhoodT.cpp         
  Compadre_OptimizationManager.cpp
  Compadre_ParameterManager.cpp
  Compadre_ParticlesT.cpp
  Compadre_PhysicsT.cpp
  Compadre_ProblemExplicitTransientT.cpp
  Compadre_ProblemT.cpp # requires linear solvers
  Compadre_RemapManager.cpp
  Compadre_RemoteDataManager.cpp
  Compadre_SolverT.cpp
  Compadre_SourcesT.cpp
  Compadre_SphericalCoordsT.cpp
  Compadre_XyzVector.cpp
  )

add_library(compadreharness ${CompadreHarness_SOURCES})
bob_library_includes(compadreharness)

# Trilinos
target_link_libraries(compadreharness PUBLIC ${Trilinos_LIBRARIES})

# NetCDF
if(CompadreHarness_USE_Netcdf)
  target_include_directories(compadreharness PUBLIC "${Netcdf_INCLUDE_DIR}")
  target_link_libraries(compadreharness PUBLIC "${Netcdf_LIBRARIES}")
endif()
if(CompadreHarness_USE_Netcdf_MPI)
  target_include_directories(compadreharness PUBLIC "${Netcdf_MPI_INCLUDE_DIR}")
endif()

# COMPOSE
if(CompadreHarness_USE_Compose)
  add_dependencies(compadreharness Compose) # compadreharness relies on Compose
  target_link_libraries(compadreharness PUBLIC "${Compose_PREFIX}/lib/libcompose${CMAKE_SHARED_LIBRARY_SUFFIX}")
  target_include_directories(compadreharness PUBLIC  
    $<BUILD_INTERFACE:${Compose_PREFIX}/include/compose>  
    $<INSTALL_INTERFACE:include> 
  )
endif()

# Compadre Toolkit
target_link_libraries(compadreharness PUBLIC compadre)

bob_export_target(compadreharness)


bob_config_header("${CMAKE_CURRENT_BINARY_DIR}/CompadreHarness_Config.h")
set(CompadreHarness_HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}/CompadreHarness_Config.h"
  Compadre_AnalyticFunctions.hpp     
  Compadre_BoundaryConditionsT.hpp   
  Compadre_CoordsT.hpp               
  Compadre_DOFData.hpp               
  Compadre_DOFManager.hpp            
  Compadre_EuclideanCoordsT.hpp      
  Compadre_FieldManager.hpp          
  Compadre_FieldT.hpp                
  Compadre_FileIO.hpp                
  Compadre_GlobalConstants.hpp       
  Compadre_LagCoordsT.hpp            
  Compadre_MultiJumpNeighborhood.hpp         
  Compadre_nanoflannPointCloudT.hpp
  Compadre_NeighborhoodT.hpp         
  Compadre_OptimizationManager.hpp                 
  Compadre_ParameterManager.hpp      
  Compadre_ParticlesT.hpp
  Compadre_PhysicsT.hpp
  Compadre_ProblemExplicitTransientT.hpp
  Compadre_ProblemT.hpp
  Compadre_RemapManager.hpp
  Compadre_RemoteDataManager.hpp
  Compadre_SimpleCoords.hpp
  Compadre_SimpleEuclidean.hpp
  Compadre_SimpleMPI.hpp
  Compadre_SimpleSphericalCoords.hpp
  Compadre_SolverT.hpp
  Compadre_SourcesT.hpp
  Compadre_SphericalCoordsT.hpp
  CompadreHarness_Typedefs.hpp
  Compadre_XyzVector.hpp
  )

set(CompadreHarness_TPL_Cobra
  TPL/Cobra_SLBQP.hpp
  )
  

install(FILES ${CompadreHarness_HEADERS} DESTINATION include)
install(FILES ${CompadreHarness_TPL_Cobra} DESTINATION include/TPL)

bob_end_subdir()


