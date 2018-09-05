
# everything requires Tpetra, Zoltan2, TeuchosComm, and TeuchosParameterList
# ProblemT requires linear solvers, i.e. SolverT.cpp
# SolveT requires 
set(Compadre_SOURCES
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
  Compadre_nanoflannInformation.cpp  
  Compadre_nanoflannPointCloudT.cpp  
  Compadre_NeighborhoodT.cpp         
  Compadre_OBFET.cpp
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

add_library(compadre ${Compadre_SOURCES})
bob_library_includes(compadre)


if (Compadre_USE_Trilinos)
   target_link_libraries(compadre PUBLIC ${Trilinos_LIBRARIES})
endif()

if (Compadre_USE_VTK)
  target_link_libraries(compadre PUBLIC "${VTK_LIBRARIES}")
endif()

if(Compadre_USE_Netcdf)
  target_include_directories(compadre PUBLIC "${Netcdf_INCLUDE_DIR}")
  target_link_libraries(compadre PUBLIC "${Netcdf_LIBRARIES}")
endif()
if(Compadre_USE_Netcdf_MPI)
  target_include_directories(compadre PUBLIC "${Netcdf_MPI_INCLUDE_DIR}")
endif()



target_link_libraries(compadre PUBLIC gmls)

bob_export_target(compadre)


bob_config_header("${CMAKE_CURRENT_BINARY_DIR}/Compadre_Config.h")
set(Compadre_HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}/Compadre_Config.h"
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
  Compadre_nanoflannInformation.hpp  
  Compadre_nanoflannPointCloudT.hpp  
  Compadre_NeighborhoodT.hpp         
  Compadre_OBFET.hpp                 
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
  Compadre_Typedefs.hpp
  Compadre_XyzVector.hpp
  )


install(FILES ${Compadre_HEADERS} DESTINATION include)

bob_end_subdir()


