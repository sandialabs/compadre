cmake_minimum_required(VERSION 3.0.0)

##########
#
#   Compadre Details
#
########## 
PROJECT(Compadre VERSION 0.0.1 LANGUAGES CXX)

# cmake helper functions
include(TPLs/cmake_bob/bob.cmake)

bob_begin_package()




##########
#
#
#  TPLS
#
#
##########




#NANOFLANN
bob_option(Compadre_USE_Nanoflann "Use Nanoflann" ON) # header included in TPLs

#BOOST
bob_option(Compadre_USE_Boost "Use Boost" ON)
if (Compadre_USE_Boost)
  find_path(Boost_INCLUDE_DIR NAMES "boost/wave.hpp" HINT ${Boost_PREFIX})
  set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR})
  MESSAGE(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIR}")
  IF (NOT(Boost_INCLUDE_DIRS))
    MESSAGE(FATAL_ERROR "Compadre_USE_Boost is ON, but Boost_PREFIX was set incorrectly.")
  ENDIF()
endif()

#TRILINOS
bob_option(Compadre_USE_Trilinos "Use Trilinos" ON)
bob_input(Trilinos_PREFIX "" PATH "Path to Trilinos install")
bob_add_dependency(PUBLIC NAME Trilinos
    TARGETS trilinos
    INCLUDE_DIR_VARS
      Trilinos_INCLUDE_DIRS
      Trilinos_TPL_INCLUDE_DIRS
LIBRARY_VARS Trilinos_LIBRARIES Trilinos_TPL_LIBRARIES)
SET(TRILINOS_LINEAR_SOLVES OFF)
IF (Trilinos_FOUND)
	LIST(REVERSE Trilinos_LIBRARIES)
	LIST(REMOVE_DUPLICATES Trilinos_LIBRARIES)
	LIST(REVERSE Trilinos_LIBRARIES)

	LIST(REVERSE Trilinos_TPL_LIBRARIES)
	LIST(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
	LIST(REVERSE Trilinos_TPL_LIBRARIES)
	
	MESSAGE("\nFound Trilinos!  Here are the details: ")
	MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
	MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
	MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
	MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
	MESSAGE("   Trilinos_BIN_DIRS = ${Trilinos_BIN_DIRS}")
	MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
	MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
	MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
	MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
	MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
	MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
	MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
	MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
	MESSAGE("End of Trilinos details\n")
	
	LIST(REVERSE Trilinos_INCLUDE_DIRS)
	LIST(REMOVE_DUPLICATES Trilinos_INCLUDE_DIRS)
	LIST(REVERSE Trilinos_INCLUDE_DIRS)
	MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
	MESSAGE("   Trilinos_CXX_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
	
	LIST(FIND Trilinos_PACKAGE_LIST Zoltan2 Zoltan2TrilinosID)

	LIST(FIND Trilinos_PACKAGE_LIST Belos BelosTrilinosID)
	LIST(FIND Trilinos_PACKAGE_LIST Ifpack2 Ifpack2TrilinosID)
	LIST(FIND Trilinos_PACKAGE_LIST Amesos2 Amesos2TrilinosID)
	LIST(FIND Trilinos_PACKAGE_LIST MueLu MueLuTrilinosID)
	LIST(FIND Trilinos_PACKAGE_LIST Thyra ThyraTrilinosID)
	LIST(FIND Trilinos_PACKAGE_LIST Teko TekoTrilinosID)
	LIST(FIND Trilinos_PACKAGE_LIST Stratimikos StratimikosTrilinosID)
	IF (Zoltan2TrilinosID GREATER -1 )
		MESSAGE(STATUS "Found Zoltan2 inside Trilinos!")
	ELSE()
		MESSAGE(STATUS "Found Trilinos but could not find Zoltan2.")
	ENDIF()


	# LINEAR SOLVER RELATED PACKAGES, ITERATIVE AND DIRECT
	IF ((BelosTrilinosID GREATER -1 ))
		MESSAGE(STATUS "Found Belos inside Trilinos!")
	ELSE()
		MESSAGE(STATUS "Found Trilinos but could not find Belos.")
	ENDIF()
	IF (Ifpack2TrilinosID GREATER -1 )
		MESSAGE(STATUS "Found Ifpack2 inside Trilinos!")
	ELSE()
		MESSAGE(STATUS "Found Trilinos but could not find Ifpack2.")
        ENDIF()
	IF (Amesos2TrilinosID GREATER -1 )
		MESSAGE(STATUS "Found Amesos2 inside Trilinos!")
	ELSE()
		MESSAGE(STATUS "Found Trilinos but could not find Amesos2.")
        ENDIF()
	IF (MueLuTrilinosID GREATER -1 )
		MESSAGE(STATUS "Found MueLu inside Trilinos!")
	ELSE()
		MESSAGE(STATUS "Found Trilinos but could not find MueLu.")
        ENDIF()

	IF (ThyraTrilinosID GREATER -1 )
		MESSAGE(STATUS "Found Thyra inside Trilinos!")
	ELSE()
		MESSAGE(STATUS "Found Trilinos but could not find Thyra.")
        ENDIF()

	IF (TekoTrilinosID GREATER -1 )
		MESSAGE(STATUS "Found Teko inside Trilinos!")
	ELSE()
		MESSAGE(STATUS "Found Trilinos but could not find Teko.")
        ENDIF()
	IF (StratimikosTrilinosID GREATER -1 )
		MESSAGE(STATUS "Found Stratimikos inside Trilinos!")
	ELSE()
		MESSAGE(STATUS "Found Trilinos but could not find Stratimikos.")
        ENDIF()

	IF ((Amesos2TrilinosID GREATER -1 ) AND (BelosTrilinosID GREATER -1 ) AND (Ifpack2TrilinosID GREATER -1 ) AND (MueLuTrilinosID GREATER -1 ) AND (ThyraTrilinosID GREATER -1 ) AND (TekoTrilinosID GREATER -1 ) AND (StratimikosTrilinosID GREATER -1 ))
		MESSAGE(STATUS "All packages needed for linear solves in Trilinos found.")
		SET(TRILINOS_LINEAR_SOLVES ON)
        ENDIF()
ENDIF()



#KOKKOS
set(Compadre_USE_KokkosCore_DEFAULT ON)
set(KokkosCore_PREFIX_DEFAULT "${Trilinos_PREFIX}")
bob_add_dependency(PUBLIC NAME KokkosCore TARGETS kokkoscore INCLUDE_DIR_VARS KokkosCore_INCLUDE_DIRS)

#NETCDF
bob_option(Compadre_USE_Netcdf "Use Netcdf" ON)
bob_input(Netcdf_PREFIX "" PATH "Path to Netcdf install")
#set(Netcdf_PREFIX_DEFAULT "")
if (Compadre_USE_Netcdf)
  find_path(Netcdf_INCLUDE_DIR NAMES "netcdf.h" HINT ${Netcdf_PREFIX})
  find_library(Netcdf_LIBRARY netcdf HINT "${Netcdf_INCLUDE_DIR}/../lib" NO_DEFAULT_PATH)
  set(Netcdf_INCLUDE_DIRS ${Netcdf_INCLUDE_DIR})
  set(Netcdf_LIBRARIES ${Netcdf_LIBRARY})
  MESSAGE(STATUS "Netcdf_INCLUDE_DIRS: ${Netcdf_INCLUDE_DIR}")
  MESSAGE(STATUS "Netcdf_LIBRARIES: ${Netcdf_LIBRARY}")
  IF (NOT(Netcdf_INCLUDE_DIRS) OR NOT(Netcdf_LIBRARIES))
    MESSAGE(FATAL_ERROR "Compadre_USE_Netcdf is ON, but Netcdf_PREFIX was set incorrectly.")
  ENDIF()
  find_path(Netcdf_MPI_INCLUDE_DIR NAMES "netcdf_par.h" HINT ${Netcdf_PREFIX})
  if (NOT(Netcdf_MPI_INCLUDE_DIR))
    MESSAGE(STATUS "Netcdf with MPI_IO support NOT found.")
    set(Compadre_USE_Netcdf_MPI OFF)
  else()
    MESSAGE(STATUS "Netcdf with MPI_IO support found.")
    set(Compadre_USE_Netcdf_MPI ON)
  endif()
  MESSAGE(STATUS "Netcdf_MPI_INCLUDE_DIRS: ${Netcdf_MPI_INCLUDE_DIR}")
else()
    set(Compadre_USE_Netcdf_MPI OFF)
endif()
   
#VTK
bob_option(Compadre_USE_VTK "Use VTK" ON)
bob_input(VTK_PREFIX "" PATH "Path to VTK install")
IF (Compadre_USE_VTK)
  SET(CMAKE_PREFIX_PATH ${VTK_PREFIX} ${CMAKE_PREFIX_PATH})
  FIND_PACKAGE(VTK)
  IF (VTK_FOUND)
        SET(VTK_DIR "${VTK_PREFIX}")
  	INCLUDE(${VTK_USE_FILE})
  	MESSAGE("\nFound VTK!  Here are the details: ")
  	MESSAGE("   VTK_DIR = ${VTK_DIR}")
  	MESSAGE("   VTK_VERSION = ${VTK_VERSION}")
  	MESSAGE("   VTK_LIBRARIES = ${VTK_LIBRARIES}")
  	MESSAGE("   VTK_LIBRARY_DIRS = ${VTK_LIBRARY_DIRS}")
  
  	LIST(FIND VTK_LIBRARIES vtkIOMPIParallel HAVE_VTKMPIIO)
  	IF (HAVE_VTKMPIIO GREATER -1 )
    		MESSAGE(STATUS "VTK with MPI_IO support found.")
  	ELSE()
  		MESSAGE(STATUS "VTK found, but not built with MPI support.\n Try rebuilding with VTK with:-D VTK_USE_PARALLEL:BOOL=ON, -D VTK_USE_MPI:BOOL=ON, -D Module_vtkFiltersParallelMPI:BOOL=ON, -D Module_vtkIOMPIParallel:BOOL=ON \n")
  	ENDIF()
  ELSE()
    MESSAGE(FATA_ERROR "VTK not found, but requested because Compadre_USE_VTK=ON")
  ENDIF()
ENDIF()

#MPI
FIND_PACKAGE(MPI)
bob_option(Compadre_USE_MPI "Use MPI for parallelism" ${MPI_FOUND})



##########
#
#
#  OPTIONS
#
#
##########


bob_option(GMLS_TESTS "Compile GMLS tests" ON)
bob_option(Compadre_TESTS "Compile Compadre tests" ON)

set(GMLS_EXAMPLES_DEFAULT "${GMLS_TESTS}")
set(Compadre_EXAMPLES_DEFAULT "${Compadre_TESTS}")

# physics compiled by default if Compadre_EXAMPLES is ON and not if OFF
bob_option(Compadre_PHYSICS "Compile Compadre Physics" "${Compadre_EXAMPLES}")


bob_option(Compadre_CANGA_ENABLED "Compile optional CANGA tests and examples" OFF)

##########
#
#
#  Kokkos Configurations
#
#
##########


##  Detect KokkosCore functionality from Trilinos
include(${CMAKE_CURRENT_LIST_DIR}/TPLs/cmake_bob/detect_trilinos_opts.cmake)
detect_trilinos_opts()
if (KokkosCore_HAS_CUDA AND (NOT KokkosCore_HAS_CUDA_LAMBDA))
  message(FATAL_ERROR
          "Please reconfigure Kokkos with -DKokkos_ENABLE_Cuda_Lambda:BOOL=ON")
endif()


##
##   OpenMP / Cuda
##
if (Compadre_USE_CUDA)
  find_package(CUDA)
  bob_input(Compadre_CUDA_ARCH "" STRING "CUDA arch specifier")
endif()
bob_option(Compadre_USE_OpenMP "Whether to use OpenMP" "${KokkosCore_HAS_OpenMP}")
#bob_option(Compadre_USE_CUDA "Whether to use CUDA" "${KokkosCore_HAS_CUDA}")



##########
#
#
#  CXX FLAGS
#
#
##########




bob_begin_cxx_flags()
bob_cxx11_flags()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(FLAGS "${FLAGS} -fno-omit-frame-pointer")
  if(Compadre_USE_CUDA)
    if (Compadre_CUDA_ARCH)
      add_compile_options($<$<COMPILE_LANGUAGE:CXX>:--cuda-gpu-arch=${Compadre_CUDA_ARCH}>)
    endif()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-x>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:cuda>)
    # -Wweak-vtables gives false positives with -x cuda
    set(FLAGS "${FLAGS} -Wno-weak-vtables")
  endif()
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  if(Compadre_USE_CUDA)
    set(FLAGS "${FLAGS} -expt-extended-lambda -lineinfo")
    if (Compadre_CUDA_ARCH)
      set(FLAGS "${FLAGS} -arch=${Compadre_CUDA_ARCH}")
    endif()
    if (CUDA_VERSION EQUAL 9)
      set(FLAGS "${FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    endif()
  else()
    if(Compadre_USE_Boost)
      set(FLAGS "${FLAGS} -fext-numeric-literals")
    endif()
    set(FLAGS "${FLAGS} -fno-omit-frame-pointer")
    if(Compadre_USE_OpenMP)
      set(FLAGS "${FLAGS} -fopenmp")
    endif()
  endif()
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  set(FLAGS "${FLAGS} -fno-omit-frame-pointer")
  if(Compadre_USE_OpenMP)
    set(FLAGS "${FLAGS} -fopenmp")
  endif()
else()
  message(WARNING "Unexpected compiler type ${CMAKE_CXX_COMPILER_ID}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")



## Get compiler flags from Trilinos
#IF (Trilinos_CXX_COMPILER_FLAGS) 
#  FOREACH(arg ${Trilinos_CXX_COMPILER_FLAGS} )
#    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${arg}")
#  ENDFOREACH()
#ENDIF()

bob_end_cxx_flags()




##########
#
#
#  PREPROCESSOR DEFINES FOR USE WHEN BUILDING CODE
#
#
##########



set(Compadre_KEY_BOOLS
    Compadre_USE_MPI
    Compadre_USE_Boost
    Compadre_USE_KokkosCore
    Compadre_USE_OpenMP
    Compadre_USE_CUDA
    Compadre_USE_Trilinos
    Compadre_USE_Nanoflann
    Compadre_USE_Netcdf
    Compadre_USE_Netcdf_MPI
    Compadre_USE_VTK
    TRILINOS_LINEAR_SOLVES
    Compadre_CANGA_ENABLED
   )

set(Compadre_KEY_INTS
    Compadre_VERSION_MAJOR
    Compadre_VERSION_MINOR
    Compadre_VERSION_PATCH
   )

set(Compadre_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(Compadre_KEY_STRINGS
    Compadre_SEMVER
    Compadre_COMMIT
    Compadre_CXX_FLAGS
    Compadre_CMAKE_ARGS
)



bob_get_commit()
bob_get_semver()




##########
#
#
#  GMLS AND COMPADRE LIBRARIES
#
#
##########


add_subdirectory(GMLS)
add_subdirectory(src)

if(Compadre_PHYSICS)
  add_subdirectory(physics)
endif()

if(GMLS_EXAMPLES OR Compadre_EXAMPLES)
  add_subdirectory(examples)
#  add_subdirectory(tests)
endif()


bob_end_package()








#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#INCLUDE_DIRECTORIES(.)
#SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
#SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
#SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
#INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}/src")
#IF (NOT APPLE)
#  SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals" ) 
#ENDIF()
#IF (BUILD_SHARED_LIBRARIES)
#  SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" ) 
#ENDIF()
#
##
##       Manage TPLs
##
#INCLUDE("./TPLs/CMakeLists.txt")
#
##
##       includes gsl libraries
##       builds gmls library
##
#ADD_SUBDIRECTORY(GMLS)
#SET(GMLS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/GMLS/src")
#
#
#IF (Trilinos_FOUND AND MPI_FOUND)
##
##       Build COMPADRE library
##
#FILE(GLOB COMPADRE_CPPFILES ./src/*.cpp ./src/problems/*.cpp)
#
#IF(BUILD_SHARED_COMPADRE_LIBRARY)
#  ADD_LIBRARY(COMPADRE_LIBRARIES SHARED ${COMPADRE_CPPFILES})
#ELSE()
#  ADD_LIBRARY(COMPADRE_LIBRARIES STATIC ${COMPADRE_CPPFILES})
#ENDIF(BUILD_SHARED_COMPADRE_LIBRARY)
#
#TARGET_INCLUDE_DIRECTORIES(COMPADRE_LIBRARIES PUBLIC ${Trilinos_INCLUDE_DIRS} ${GMLS_INCLUDE_DIRS})
#TARGET_LINK_LIBRARIES(COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ${GMLS_LIBRARIES})
#
#IF (USE_VTK)
#  TARGET_LINK_LIBRARIES(COMPADRE_LIBRARIES ${VTK_LIBRARIES})
#ENDIF()
#
#IF (USE_NETCDF)
#  TARGET_LINK_LIBRARIES(COMPADRE_LIBRARIES ${HDF5_LIB} ${HDF5_HL_LIB} ${CURL_LIB} ${NETCDF_LIB})
#ENDIF()
#
#IF(USE_BOOST)
#  TARGET_INCLUDE_DIRECTORIES(COMPADRE_LIBRARIES PUBLIC ${BOOST_INC})
#ENDIF()
#
#SET(COMPADRE_INCLUDE_DIRS src)
#INCLUDE_DIRECTORIES(${COMPADRE_INCLUDE_DIRS})
#INCLUDE_DIRECTORIES(${COMPADRE_INCLUDE_DIRS}/problems)
#
## Get compiler flags from Trilinos
#IF (Trilinos_CXX_COMPILER_FLAGS) 
#  FOREACH(arg ${Trilinos_CXX_COMPILER_FLAGS} )
#    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${arg}")
#  ENDFOREACH()
#ENDIF()
#
#
## Canga Enabled Flags
#IF (NOT DEFINED CANGA_ENABLED)
#  SET(CANGA_ENABLED OFF) # on by default
#ENDIF()
#
#
##
##	Executables
##
#ADD_EXECUTABLE(GMLS_Laplacian_Test.exe CompadreGMLSTest.cpp)
#TARGET_LINK_LIBRARIES(GMLS_Laplacian_Test.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(tpetraCoordsTest.exe CompadreTrilinosTest.cpp)
#TARGET_LINK_LIBRARIES(tpetraCoordsTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(kdtreeTest.exe Compadre_KdTreeTest.cpp)
#TARGET_LINK_LIBRARIES(kdtreeTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(readerTest.exe Compadre_SphReaderTest.cpp)
#TARGET_LINK_LIBRARIES(readerTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(parallelReader.exe Compadre_ParallelReaderTest.cpp)
#TARGET_LINK_LIBRARIES(parallelReader.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(spherePoisson.exe Compadre_SpherePoissonTest.cpp)
#TARGET_LINK_LIBRARIES(spherePoisson.exe COMPADRE_LIBRARIES GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(simpleCoordsTest.exe Compadre_SimpleCoordsUnitTest.cpp)
#TARGET_LINK_LIBRARIES(simpleCoordsTest.exe COMPADRE_LIBRARIES)
#
#ADD_EXECUTABLE(simpleMPITest.exe Compadre_SimpleMPIUnitTest.cpp)
#TARGET_LINK_LIBRARIES(simpleMPITest.exe COMPADRE_LIBRARIES)
#
#ADD_EXECUTABLE(remapTest.exe Compadre_RemapTest.cpp)
#TARGET_LINK_LIBRARIES(remapTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(lagrangianTest.exe Compadre_LagrangianTest.cpp)
#TARGET_LINK_LIBRARIES(lagrangianTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(remoteRemapTest.exe Compadre_RemoteRemapTest.cpp)
#TARGET_LINK_LIBRARIES(remoteRemapTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#IF(CANGA_ENABLED)
#  ADD_EXECUTABLE(cangaRemoteRemap.exe Canga_RemoteRemap.cpp)
#  TARGET_LINK_LIBRARIES(cangaRemoteRemap.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#  ADD_EXECUTABLE(cangaIntercomparison.exe Canga_Intercomparison.cpp)
#  TARGET_LINK_LIBRARIES(cangaIntercomparison.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#ENDIF()
#
#ADD_EXECUTABLE(lagrangianShallowWater.exe Compadre_LagrangianShallowWater.cpp)
#TARGET_LINK_LIBRARIES(lagrangianShallowWater.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(laplaceBeltrami.exe Compadre_LaplaceBeltrami.cpp)
#TARGET_LINK_LIBRARIES(laplaceBeltrami.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(textToCoefficientReaderWriter.exe textToCoefficientReaderWriter.cpp)
#TARGET_LINK_LIBRARIES(textToCoefficientReaderWriter.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
##
##
##	Custom build targets
##
#ADD_CUSTOM_TARGET(compadre)
#ADD_DEPENDENCIES(compadre COMPADRE_LIBRARIES)
#
#ADD_CUSTOM_TARGET(unit)
#ADD_DEPENDENCIES(unit simpleCoordsTest.exe simpleMPITest.exe)
#
#ADD_CUSTOM_TARGET(regression)
#ADD_DEPENDENCIES(regression tpetraCoordsTest.exe kdtreeTest.exe readerTest.exe spherePoisson.exe)
#
#ADD_CUSTOM_TARGET(gmls)
#ADD_DEPENDENCIES(gmls GMLS_Laplacian_Test.exe)
#
#ADD_CUSTOM_TARGET(parallel)
#ADD_DEPENDENCIES(parallel parallelReader.exe)
#
#ADD_CUSTOM_TARGET(remap)
#ADD_DEPENDENCIES(remap remapTest.exe)
#
#ADD_CUSTOM_TARGET(shallow)
#ADD_DEPENDENCIES(shallow lagrangianShallowWater.exe)
#
#IF(CANGA_ENABLED)
#  ADD_CUSTOM_TARGET(canga)
#  ADD_DEPENDENCIES(canga cangaRemoteRemap.exe)
#  ADD_DEPENDENCIES(canga cangaIntercomparison.exe)
#ENDIF()
#
#endif()
#
#
##
##	Testing
##
### When executing these with OMP, it makes sense to add --bind-to-none or --bind-to none in order to not restrict threads to cores
#ENABLE_TESTING()
#ADD_SUBDIRECTORY(tests)
#
#CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/src/Compadre_Config.h.in" "${PROJECT_BINARY_DIR}/src/Compadre_Config.h")
#CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/tests/parameters_amg.xml.in" "${CMAKE_SOURCE_DIR}/tests/trilinos_test/parameters_amg.xml" @ONLY)
#CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/tests/parameters_amg_blockdiag5x5.xml.in" "${CMAKE_SOURCE_DIR}/tests/trilinos_test/parameters_amg_blockdiag5x5.xml" @ONLY)

