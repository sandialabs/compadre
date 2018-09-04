
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
  Compadre_ProblemT.cpp
  Compadre_RemapManager.cpp
  Compadre_RemoteDataManager.cpp
  Compadre_SolverT.cpp
  Compadre_SourcesT.cpp
  Compadre_SphericalCoordsT.cpp
  Compadre_VTKInformation.cpp
  Compadre_XyzVector.cpp
  Compadre_AnalyticFunctions.cpp
  Compadre_CoordsT.cpp
  )

add_library(compadre ${Compadre_SOURCES})
bob_library_includes(compadre)





#bob_link_dependency(compadre PUBLIC KokkoScore)
if (Compadre_USE_KokkosCore)
  target_include_directories(compadre PUBLIC "${KokkosCore_INCLUDE_DIRS}")
endif()

bob_link_dependency(compadre PRIVATE Trilinos::TeuchosComm)
#bob_link_dependency(compadre PRIVATE Trilinos::TeuchosComm)
#if (Compadre_USE_Trilinos)
#  target_include_directories(compadre PRIVATE "${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS}")
#  target_link_libraries(compadre PRIVATE "${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}")
#endif()

if (Compadre_USE_VTK)
  #bob_link_dependency(compadre PUBLIC *)
  #bob_link_dependency(compadre PUBLIC vtkFiltersParallelDIY2 vtkFiltersParallelMPI vtkIOMPIImage vtkIOMPIParallel vtkIOParallel vtkParallelMPI vtkIOParallelXML vtkParallelCore vtkIOLegacy vtkIOXML vtkIOCore vtkFiltersGeneral vtkFiltersCore vtkCommonExecutionModel vtkCommonDataModel vtkCommonCore)

  #target_link_libraries(compadre PUBLIC vtkFiltersParallelDIY2 vtkFiltersParallelMPI vtkIOMPIImage vtkIOMPIParallel vtkIOParallel vtkParallelMPI vtkIOParallelXML vtkParallelCore vtkIOLegacy vtkIOXML vtkIOCore vtkFiltersGeneral vtkFiltersCore vtkCommonExecutionModel vtkCommonDataModel vtkCommonCore)
  #bob_link_dependency(compadre PUBLIC VTK)
  #target_include_directories(compadre PUBLIC "${VTK_INCLUDE_DIRS}")
  target_link_libraries(compadre PUBLIC "${VTK_LIBRARIES}")
  #target_link_libraries(compadre PUBLIC "${VTK_LIBRARY_DIRS}")
endif()

if(Compadre_USE_Netcdf)
  target_include_directories(compadre PUBLIC "${Netcdf_INCLUDE_DIR}")
  target_link_libraries(compadre PUBLIC "${Netcdf_LIBRARIES}")
endif()
if(Compadre_USE_Netcdf_MPI)
  target_include_directories(compadre PUBLIC "${Netcdf_MPI_INCLUDE_DIR}")
endif()




target_link_libraries(compadre PUBLIC gmls)



#target_link_libraries(compadre PRIVATE gmls)

#bob_link_dependency(compadre PUBLIC Compdare::gmls)
#$target_include_directories(compadre PUBLIC "../GMLS/src")

#include_directories(${GMLS_HEADERS})
#message(STATUS "GMLS_HEADERS: ${GMLS_HEADERS}")
#target_include_directories(compadre PUBLIC "${GMLS_HEADERS}")


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
  Compadre_VTKInformation.hpp
  Compadre_XyzVector.hpp
  )

bob_export_target(compadre)

install(FILES ${Compadre_HEADERS} DESTINATION include)

bob_end_subdir()


