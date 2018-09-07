cmake_minimum_required(VERSION 3.0.0)

##########
#
#   Compadre Details
#
########## 
PROJECT(Compadre VERSION 0.0.1 LANGUAGES CXX)

# cmake helper functions
include(${CMAKE_CURRENT_LIST_DIR}/cmake/bob.cmake)

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
bob_input(Boost_PREFIX "" PATH "Path to Boost install")
if (Boost_PREFIX)
  set(CMAKE_PREFIX_PATH ${Boost_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
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
if (Trilinos_PREFIX)
  set(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
bob_option(Compadre_USE_Trilinos_Solvers "Use Trilinos Linear Algebra capabilities" ON)
bob_option(Compadre_USE_Trilinos_CXX_Flags "Use Trilinos CXX flags" ON)

SET(TRILINOS_LINEAR_SOLVES OFF)
IF (Compadre_USE_Trilinos)
  FIND_PACKAGE(Trilinos)
  if (Compadre_USE_Trilinos_Solvers)
    bob_add_dependency(PUBLIC NAME Trilinos
      TARGETS "${Trilinos_LIBRARIES}"
      INCLUDE_DIR_VARS 
        Trilinos_INCLUDE_DIRS 
        Trilinos_TPL_INCLUDE_DIRS 
      LIBRARY_VARS
    Trilinos_LIBRARIES Trilinos_TPL_LIBRARIES)
  else()
    bob_add_dependency(PUBLIC NAME Trilinos
      TARGETS tpetra;kokkostsqr;tpetraclassicnodeapi;tpetraclassic;triutils;zoltan;rtop;tpetrakernels;teuchoskokkoscomm;teuchoskokkoscompat;teuchosremainder;teuchosnumerics;teuchoscomm;teuchosparameterlist;teuchoscore;kokkosalgorithms;kokkoscontainers;kokkoscore
      INCLUDE_DIR_VARS 
        Trilinos_INCLUDE_DIRS 
        Trilinos_TPL_INCLUDE_DIRS 
      LIBRARY_VARS
    Trilinos_LIBRARIES Trilinos_TPL_LIBRARIES)
  endif()
  LIST(REVERSE Trilinos_INCLUDE_DIRS)
  LIST(REMOVE_DUPLICATES Trilinos_INCLUDE_DIRS)
  LIST(REVERSE Trilinos_INCLUDE_DIRS)

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
  
  if (Compadre_USE_Trilinos_Solvers)
    IF ((Amesos2TrilinosID GREATER -1 ) AND (BelosTrilinosID GREATER -1 ) AND (Ifpack2TrilinosID GREATER -1 ) AND (MueLuTrilinosID GREATER -1 ) AND (ThyraTrilinosID GREATER -1 ) AND (TekoTrilinosID GREATER -1 ) AND (StratimikosTrilinosID GREATER -1 ))
    	MESSAGE(STATUS "All packages needed for linear solves in Trilinos found.")
    	SET(TRILINOS_LINEAR_SOLVES ON)
    ELSE()
      MESSAGE(FATAL_ERROR "Compadre_USE_Trilinos_Solvers=ON but not all packages needed in Trilinos for linear solves were found.")
    ENDIF()
  else()
    IF ((Amesos2TrilinosID GREATER -1 ) AND (BelosTrilinosID GREATER -1 ) AND (Ifpack2TrilinosID GREATER -1 ) AND (MueLuTrilinosID GREATER -1 ) AND (ThyraTrilinosID GREATER -1 ) AND (TekoTrilinosID GREATER -1 ) AND (StratimikosTrilinosID GREATER -1 ))
    	MESSAGE(STATUS "All packages needed for linear solves in Trilinos found, but not enabled because Compadre_USE_Trilinos_Solvers=OFF.")
    	SET(TRILINOS_LINEAR_SOLVES OFF)
    ENDIF()

  endif()
ENDIF()



#KOKKOS
bob_option(Compadre_USE_KokkosCore "Use KokkosCore" ON)
bob_input(KokkosCore_PREFIX "${Trilinos_PREFIX}" PATH "Path to KokkosCore install")
set(KOKKOS_IN_TRILINOS OFF)
if (KokkosCore_PREFIX)
  set(CMAKE_PREFIX_PATH ${KokkosCore_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
if (Compadre_USE_KokkosCore)
  if ("${KokkosCore_PREFIX}" STREQUAL "${Trilinos_PREFIX}") 
    SET(KOKKOS_IN_TRILINOS ON)
    MESSAGE(STATUS "Using KokkosCore in Trilinos.")
    bob_add_dependency(PUBLIC NAME KokkosCore TARGETS kokkoscore INCLUDE_DIR_VARS KokkosCore_INCLUDE_DIRS)
  else()
    FIND_FILE(Kokkos_PROJECT_FULL_FILENAME name kokkos_generated_settings.cmake HINTS "${KokkosCore_PREFIX}" NO_DEFAULT_PATH)
    GET_FILENAME_COMPONENT(Kokkos_PROJECT_FILENAME ${Kokkos_PROJECT_FULL_FILENAME} NAME)
    IF(Kokkos_PROJECT_FILENAME)
      FIND_PACKAGE(KokkosCore CONFIGS "${Kokkos_PROJECT_FILENAME}") 
      IF(KokkosCore_FOUND)
        MESSAGE(STATUS "KokkosCore found at ${Kokkos_PROJECT_FULL_FILENAME}")
        GET_FILENAME_COMPONENT(KokkosCore_INSTALL_DIR ${Kokkos_PROJECT_FULL_FILENAME} DIRECTORY)
        SET(KokkosCore_INCLUDE_DIRS "${KokkosCore_INSTALL_DIR}/include")
        SET(KokkosCore_STATIC_LIB "${KokkosCore_INSTALL_DIR}/lib/${KOKKOS_LINK_DEPENDS}")
        MESSAGE(STATUS "KokkosCore_INCLUDE_DIRS: ${KokkosCore_INCLUDE_DIRS}")
        MESSAGE(STATUS "KokkosCore_STATIC_LIB: ${KokkosCore_STATIC_LIB}")
      ELSE()
          MESSAGE(FATAL_ERROR "KokkosCore from outside of Trilinos requested, but kokkos_generated_settings.cmake was not found.")
      ENDIF()
    ELSE()
        MESSAGE(FATAL_ERROR "KokkosCore from outside of Trilinos requested, but kokkos_generated_settings.cmake was not found.")
    ENDIF()
  endif()
endif()

#NETCDF
bob_option(Compadre_USE_Netcdf "Use Netcdf" ON)
bob_input(Netcdf_PREFIX "" PATH "Path to Netcdf install")
if (Netcdf_PREFIX)
  set(CMAKE_PREFIX_PATH ${Netcdf_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
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
if (VTK_PREFIX)
  set(CMAKE_PREFIX_PATH ${VTK_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
IF (Compadre_USE_VTK)
  FIND_PACKAGE(VTK)
  bob_add_dependency(PUBLIC NAME VTK
    TARGETS "${VTK_LIBRARIES}"
    INCLUDE_DIR_VARS 
      VTK_INCLUDE_DIRS 
    LIBRARY_VARS
  VTK_LIBRARIES)
  LIST(FIND VTK_LIBRARIES vtkIOMPIParallel HAVE_VTKMPIIO)
  IF (HAVE_VTKMPIIO GREATER -1 )
  	MESSAGE(STATUS "VTK with MPI_IO support found.")
  ELSE()
  	MESSAGE(STATUS "VTK found, but not built with MPI support.\n Try rebuilding with VTK with:-D VTK_USE_PARALLEL:BOOL=ON, -D VTK_USE_MPI:BOOL=ON, -D Module_vtkFiltersParallelMPI:BOOL=ON, -D Module_vtkIOMPIParallel:BOOL=ON \n")
  ENDIF()
ENDIF()

#MPI
FIND_PACKAGE(MPI)
bob_option(Compadre_USE_MPI "Use MPI for parallelism" ${MPI_FOUND})


#PYTHON
bob_option(Compadre_USE_PYTHON "Use PYTHON" OFF)
bob_input(PYTHON_PREFIX "" PATH "Path to PYTHON install")
if (PYTHON_PREFIX)
  set(CMAKE_PREFIX_PATH ${PYTHON_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
if (Compadre_USE_PYTHON)

  FIND_PACKAGE(PythonLibs)
  IF(PythonLibs_FOUND)
    MESSAGE(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
    MESSAGE(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
    
    bob_input(PYTHON_EXECUTABLE "" PATH "Python executable location")
    IF(NOT(PYTHON_EXECUTABLE))
      MESSAGE(STATUS "Python executable location PYTHON_EXECUTABLE not given. Search made using 'which python'")
      EXECUTE_PROCESS(
        COMMAND which "python"
        OUTPUT_VARIABLE PYTHON_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE )
    ENDIF()
    MESSAGE(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
    
    EXECUTE_PROCESS(
      COMMAND "${PYTHON_EXECUTABLE}" -c "import site; print(site.USER_SITE)"
      OUTPUT_VARIABLE PYTHON_SITEPACKAGES
      OUTPUT_STRIP_TRAILING_WHITESPACE )
    
    IF(NOT EXISTS ${PYTHON_SITEPACKAGES})
      EXECUTE_PROCESS(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
        OUTPUT_VARIABLE PYTHON_SITEPACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE )
    ENDIF()
    MESSAGE(STATUS "PYTHON_SITEPACKAGES: ${PYTHON_SITEPACKAGES}")
    
    bob_input(Numpy_PREFIX "${PYTHON_SITEPACKAGES}/numpy/core/include" PATH "Installation location of Numpy headers" )
    
    FIND_PATH(Numpy_INCLUDE_DIRS numpy/arrayobject.h HINTS ${Numpy_PREFIX})
    IF (Numpy_INCLUDE_DIRS)
      MESSAGE(STATUS "Numpy_INCLUDE_DIRS: ${Numpy_INCLUDE_DIRS}")
    ELSE()
      MESSAGE(FATAL_ERROR "Numpy headers not found, but needed when Compadre_USE_PYTHON:BOOL=ON. Try setting Numpy_PREFIX.")
    ENDIF()

    set(SWIG_PREFIX "../GMLS/src/swig") # relative to examples folder
    if (PYTHON_VERSION_MAJOR EQUAL 3)
      set(GMLS_Module_DEST "lib/python3/dist-packages")
    else()
      set(GMLS_Module_DEST "lib/python/dist-packages")
    endif()
  ENDIF()

endif()


##########
#
#
#  OPTIONS
#
#
##########


bob_option(GMLS_TESTS "Compile GMLS tests" ON)
bob_option(Compadre_TESTS "Compile Compadre tests" ON)

bob_option(GMLS_EXAMPLES "Compile GMLS examples" "${GMLS_TESTS}")
bob_option(Compadre_EXAMPLES "Compile Compadre examples" "${Compadre_TESTS}")

# physics compiled by default if Compadre_EXAMPLES is ON and not if OFF
bob_option(Compadre_PHYSICS "Compile Compadre Physics" "${Compadre_EXAMPLES}")


bob_option(Compadre_CANGA_ENABLED "Compile optional CANGA tests and examples" OFF)
if ((Compadre_TESTS OR Compadre_EXAMPLES OR Compadre_PHYSICS) AND Compadre_USE_Trilinos)
  set(Compadre_Compadre_REQUIRED  ON)
else()
  set(Compadre_Compadre_REQUIRED  OFF)
endif()
MESSAGE(STATUS "Compadre harness required to be built: ${Compadre_Compadre_REQUIRED}")
bob_option(Compadre_USE_Compadre "Build Trilinos based harness for GMLS" ON)
if (Compadre_USE_Compadre OR Compadre_Compadre_REQUIRED)
  set(Compadre_USE_Compadre ON)
endif()

##########
#
#
#  Kokkos Configurations
#
#
##########


##  Detect KokkosCore functionality from Trilinos
IF (KOKKOS_IN_TRILINOS)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/detect_trilinos_opts.cmake)
  detect_trilinos_opts()
  if (KokkosCore_HAS_CUDA AND (NOT KokkosCore_HAS_CUDA_LAMBDA))
    message(FATAL_ERROR
            "Please reconfigure Kokkos with -DKokkos_ENABLE_Cuda_Lambda:BOOL=ON")
  endif()
ENDIF()


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




if(NOT(KOKKOS_IN_TRILINOS))
  IF (KOKKOS_CXX_FLAGS) 
    FOREACH(arg ${KOKKOS_CXX_FLAGS} )
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${arg}")
    ENDFOREACH()
  ENDIF()
else()
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
endif()


if (Compadre_USE_Trilinos_CXX_Flags)
  # Get compiler flags from Trilinos
  SET(CMAKE_CXX_FLAGS "")
  IF (Trilinos_CXX_COMPILER_FLAGS) 
    FOREACH(arg ${Trilinos_CXX_COMPILER_FLAGS} )
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${arg}")
    ENDFOREACH()
  ENDIF()
endif()

bob_end_cxx_flags()




##########
#
#
#  PREPROCESSOR DEFINES FOR USE WHEN BUILDING CODE
#
#
##########


set(Compadre_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

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
    Compadre_USE_PYTHON
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
    GMLS_Module_DEST
    Compadre_INSTALL_PREFIX
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

if(Compadre_USE_Compadre)
  add_subdirectory(src)
endif()

if(Compadre_USE_Trilinos AND Compadre_PHYSICS)
  add_subdirectory(physics)
endif()

if(GMLS_EXAMPLES OR Compadre_EXAMPLES)
  add_subdirectory(examples)
endif()

set (augmented_list "${Trilinos_LIBRARIES}" "${VTK_LIBRARIES}")
bob_end_package_no_recurse("${augmented_list}")
