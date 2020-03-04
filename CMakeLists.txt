cmake_minimum_required(VERSION 3.0.0)

##########
#
#   CompadreHarness Details
#
########## 
PROJECT(CompadreHarness VERSION 1.0.0 LANGUAGES CXX)

# cmake helper functions
include(${CMAKE_CURRENT_LIST_DIR}/cmake/bob.cmake)

bob_begin_package()

include(ExternalProject)


##########
#
#
#  TPLS
#
#
##########



##LAPACK
#bob_option(CompadreHarness_USE_LAPACK "Use LAPACK" ON)
#SET(CompadreHarness_USE_OPENBLAS OFF) # need to know if openblas is used in order to limit its ability to spawn threads
#bob_input(LAPACK_PREFIX "" PATH "Path to LAPACK install")
#if (LAPACK_PREFIX)
#  if(NOT IS_ABSOLUTE ${LAPACK_PREFIX})
#    set(LAPACK_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${LAPACK_PREFIX}")
#  endif()
#  set(CMAKE_PREFIX_PATH ${LAPACK_PREFIX} ${CMAKE_PREFIX_PATH})
#endif()
#bob_input(LAPACK_LOCATION "" PATH "Optional explicit path to exact location of LAPACK")
#bob_option(LAPACK_DECLARED_THREADSAFE "Is LAPACK threadsafe? If not, default to serial calls for solves." ON)
#if (CompadreHarness_USE_LAPACK)
#  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
#    # Intel compiler detected. Will use "-mkl" and no need to find Lapack or Blas
#  else()
#    if ("${LAPACK_LOCATION}" STREQUAL "")
#      FIND_LIBRARY(LAPACK_LIB lapack)
#    else()
#      SET(LAPACK_LIB "${LAPACK_LOCATION}")
#    endif()
#    MESSAGE(STATUS "LAPACK_LIB: ${LAPACK_LIB}")
#    IF (NOT(LAPACK_LIB))
#      MESSAGE(FATAL_ERROR "CompadreHarness_USE_LAPACK is ON, but LAPACK_PREFIX was set incorrectly.")
#    ELSE()
#      IF(APPLE)
#        execute_process(COMMAND otool -L  "${LAPACK_LIB}" OUTPUT_VARIABLE LAPACK_DEPENDENCIES) 
#      ELSE()
#        execute_process(COMMAND ldd "${LAPACK_LIB}" OUTPUT_VARIABLE LAPACK_DEPENDENCIES) 
#      ENDIF()
#      string(FIND "${LAPACK_DEPENDENCIES}" "openblas" BLAS_MATCH)
#      if ("${BLAS_MATCH}" GREATER "-1")
#        SET(CompadreHarness_USE_OPENBLAS ON)
#      endif()
#    ENDIF()
#  endif()
#endif()

#TRILINOS
bob_input(Trilinos_PREFIX "" PATH "Path to Trilinos install")
if (Trilinos_PREFIX)
  if(NOT IS_ABSOLUTE ${Trilinos_PREFIX})
    set(Trilinos_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${Trilinos_PREFIX}")
  endif()
  set(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})
else()
  message(FATAL_ERROR "Trilinos_PREFIX must be specified")
endif()
bob_option(CompadreHarness_USE_Trilinos_Solvers "Use Trilinos Linear Algebra capabilities" ON)
bob_option(CompadreHarness_USE_Trilinos_CXX_Flags "Use Trilinos CXX flags" ON)

SET(TRILINOS_LINEAR_SOLVES OFF)
FIND_PACKAGE(Trilinos)
#if (CompadreHarness_USE_Trilinos_Solvers)
#message("${Trilinos_LIBRARIES}, ${Trilinos_INCLUDE_DIRS}, ${Trilinos_TPL_INCLUDE_DIRS}, ${Trilinos_TPL_LIBRARIES}")
#bob_add_dependency(PUBLIC NAME Trilinos
#    TARGETS "${Trilinos_LIBRARIES}"
#    INCLUDE_DIR_VARS 
#      Trilinos_INCLUDE_DIRS 
#      Trilinos_TPL_INCLUDE_DIRS 
#    LIBRARY_VARS
#  Trilinos_LIBRARIES Trilinos_TPL_LIBRARIES)
#else()
#  bob_add_dependency(PUBLIC NAME Trilinos
#    TARGETS tpetra;kokkostsqr;tpetraclassicnodeapi;tpetraclassic;triutils;zoltan;rtop;tpetrakernels;teuchoskokkoscomm;teuchoskokkoscompat;teuchosremainder;teuchosnumerics;teuchoscomm;teuchosparameterlist;teuchoscore;kokkosalgorithms;kokkoscontainers;kokkoscore
#    INCLUDE_DIR_VARS 
#      Trilinos_INCLUDE_DIRS 
#      Trilinos_TPL_INCLUDE_DIRS 
#    LIBRARY_VARS
#  Trilinos_LIBRARIES Trilinos_TPL_LIBRARIES)
#endif()
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

if (CompadreHarness_USE_Trilinos_Solvers)
  IF ((Amesos2TrilinosID GREATER -1 ) AND (BelosTrilinosID GREATER -1 ) AND (Ifpack2TrilinosID GREATER -1 ) AND (MueLuTrilinosID GREATER -1 ) AND (ThyraTrilinosID GREATER -1 ) AND (TekoTrilinosID GREATER -1 ) AND (StratimikosTrilinosID GREATER -1 ))
    MESSAGE(STATUS "All packages needed for linear solves in Trilinos found.")
    SET(TRILINOS_LINEAR_SOLVES ON)
  ELSE()
    MESSAGE(FATAL_ERROR "CompadreHarness_USE_Trilinos_Solvers=ON but not all packages needed in Trilinos for linear solves were found.")
  ENDIF()
else()
  IF ((Amesos2TrilinosID GREATER -1 ) AND (BelosTrilinosID GREATER -1 ) AND (Ifpack2TrilinosID GREATER -1 ) AND (MueLuTrilinosID GREATER -1 ) AND (ThyraTrilinosID GREATER -1 ) AND (TekoTrilinosID GREATER -1 ) AND (StratimikosTrilinosID GREATER -1 ))
    MESSAGE(STATUS "All packages needed for linear solves in Trilinos found, but not enabled because CompadreHarness_USE_Trilinos_Solvers=OFF.")
    SET(TRILINOS_LINEAR_SOLVES OFF)
  ENDIF()

endif()



##KOKKOS
#bob_option(CompadreHarness_USE_KokkosCore "Use KokkosCore" ON)
#bob_input(KokkosCore_PREFIX "${Trilinos_PREFIX}" PATH "Path to KokkosCore install")
#set(KOKKOS_IN_TRILINOS OFF)
#if (KokkosCore_PREFIX)
#  if(NOT IS_ABSOLUTE ${KokkosCore_PREFIX})
#    set(KokkosCore_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${KokkosCore_PREFIX}")
#  endif()
#  set(CMAKE_PREFIX_PATH ${KokkosCore_PREFIX} ${CMAKE_PREFIX_PATH})
#endif()
#if (CompadreHarness_USE_KokkosCore)
#  if ("${KokkosCore_PREFIX}" STREQUAL "${Trilinos_PREFIX}") 
#    SET(KOKKOS_IN_TRILINOS ON)
#    MESSAGE(STATUS "Using KokkosCore in Trilinos.")
#    bob_add_dependency(PUBLIC NAME KokkosCore TARGETS kokkoscore INCLUDE_DIR_VARS KokkosCore_INCLUDE_DIRS)
#  else()
#    FIND_FILE(Kokkos_PROJECT_FULL_FILENAME name kokkos_generated_settings.cmake HINTS "${KokkosCore_PREFIX}" NO_DEFAULT_PATH)
#    GET_FILENAME_COMPONENT(Kokkos_PROJECT_FILENAME ${Kokkos_PROJECT_FULL_FILENAME} NAME)
#    IF(Kokkos_PROJECT_FILENAME)
#      FIND_PACKAGE(KokkosCore CONFIGS "${Kokkos_PROJECT_FILENAME}") 
#      IF(KokkosCore_FOUND)
#        MESSAGE(STATUS "KokkosCore found at ${Kokkos_PROJECT_FULL_FILENAME}")
#        GET_FILENAME_COMPONENT(KokkosCore_INSTALL_DIR ${Kokkos_PROJECT_FULL_FILENAME} DIRECTORY)
#        SET(KokkosCore_INCLUDE_DIRS "${KokkosCore_INSTALL_DIR}/include")
#        SET(KokkosCore_STATIC_LIB "${KokkosCore_INSTALL_DIR}/lib/${KOKKOS_LINK_DEPENDS}")
#        MESSAGE(STATUS "KokkosCore_INCLUDE_DIRS: ${KokkosCore_INCLUDE_DIRS}")
#        MESSAGE(STATUS "KokkosCore_STATIC_LIB: ${KokkosCore_STATIC_LIB}")
#      ELSE()
#          MESSAGE(FATAL_ERROR "KokkosCore from outside of Trilinos requested, but kokkos_generated_settings.cmake was not found.")
#      ENDIF()
#    ELSE()
#        MESSAGE(FATAL_ERROR "KokkosCore from outside of Trilinos requested, but kokkos_generated_settings.cmake was not found.")
#    ENDIF()
#  endif()
#else()
#  MESSAGE(FATAL_ERROR "CompadreHarness_USE_KokkosCore set to OFF, but entire toolkit relies on KokkosCore.")
#endif()

#NETCDF
bob_option(CompadreHarness_USE_Netcdf "Use Netcdf" ON)
bob_input(Netcdf_PREFIX "" PATH "Path to Netcdf install")
if (Netcdf_PREFIX)
  if(NOT IS_ABSOLUTE ${Netcdf_PREFIX})
    set(Netcdf_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${Netcdf_PREFIX}")
  endif()
  set(CMAKE_PREFIX_PATH ${Netcdf_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
if (CompadreHarness_USE_Netcdf)
  find_path(Netcdf_INCLUDE_DIR NAMES "netcdf.h" HINT ${Netcdf_PREFIX})
  find_library(Netcdf_LIBRARY netcdf HINT "${Netcdf_INCLUDE_DIR}/../lib" NO_DEFAULT_PATH)
  set(Netcdf_INCLUDE_DIRS ${Netcdf_INCLUDE_DIR})
  set(Netcdf_LIBRARIES ${Netcdf_LIBRARY})
  MESSAGE(STATUS "Netcdf_INCLUDE_DIRS: ${Netcdf_INCLUDE_DIR}")
  MESSAGE(STATUS "Netcdf_LIBRARIES: ${Netcdf_LIBRARY}")
  IF (NOT(Netcdf_INCLUDE_DIRS) OR NOT(Netcdf_LIBRARIES))
    MESSAGE(FATAL_ERROR "CompadreHarness_USE_Netcdf is ON, but Netcdf_PREFIX was set incorrectly.")
  ENDIF()
  find_path(Netcdf_MPI_INCLUDE_DIR NAMES "netcdf_par.h" HINT ${Netcdf_PREFIX})
  if (NOT(Netcdf_MPI_INCLUDE_DIR))
    MESSAGE(STATUS "Netcdf with MPI_IO support NOT found.")
    set(CompadreHarness_USE_Netcdf_MPI OFF)
  else()
    MESSAGE(STATUS "Netcdf with MPI_IO support found.")
    set(CompadreHarness_USE_Netcdf_MPI ON)
  endif()
  MESSAGE(STATUS "Netcdf_MPI_INCLUDE_DIRS: ${Netcdf_MPI_INCLUDE_DIR}")
else()
    set(CompadreHarness_USE_Netcdf_MPI OFF)
endif()
   

##MPI
#FIND_PACKAGE(MPI)
#bob_option(CompadreHarness_USE_MPI "Use MPI for parallelism" ${MPI_FOUND})
##PYTHON
#bob_option(CompadreHarness_USE_PYTHON "Use PYTHON" OFF)
#bob_input(PYTHON_PREFIX "" PATH "Path to PYTHON install")
#if (PYTHON_PREFIX)
#  if(NOT IS_ABSOLUTE ${PYTHON_PREFIX})
#    set(PYTHON_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_PREFIX}")
#  endif()
#  set(CMAKE_PREFIX_PATH ${PYTHON_PREFIX} ${CMAKE_PREFIX_PATH})
#endif()
#if (CompadreHarness_USE_PYTHON)
#
#  FIND_PACKAGE(PythonLibs)
#  IF(PythonLibs_FOUND)
#    MESSAGE(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
#    MESSAGE(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
#    
#    bob_input(PYTHON_EXECUTABLE "" PATH "Python executable location")
#    IF(NOT(PYTHON_EXECUTABLE))
#      MESSAGE(STATUS "Python executable location PYTHON_EXECUTABLE not given. Search made using 'which python'")
#      EXECUTE_PROCESS(
#        COMMAND which "python"
#        OUTPUT_VARIABLE PYTHON_EXECUTABLE
#        OUTPUT_STRIP_TRAILING_WHITESPACE )
#    ENDIF()
#    MESSAGE(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
#    
#    EXECUTE_PROCESS(
#      COMMAND "${PYTHON_EXECUTABLE}" -c "import site; print(site.USER_SITE)"
#      OUTPUT_VARIABLE PYTHON_SITEPACKAGES
#      OUTPUT_STRIP_TRAILING_WHITESPACE )
#    
#    IF(NOT EXISTS ${PYTHON_SITEPACKAGES})
#      EXECUTE_PROCESS(
#        COMMAND "${PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
#        OUTPUT_VARIABLE PYTHON_SITEPACKAGES
#        OUTPUT_STRIP_TRAILING_WHITESPACE )
#    ENDIF()
#    MESSAGE(STATUS "PYTHON_SITEPACKAGES: ${PYTHON_SITEPACKAGES}")
#    
#    bob_input(Numpy_PREFIX "${PYTHON_SITEPACKAGES}/numpy/core/include" PATH "Installation location of Numpy headers" )
#    
#    FIND_PATH(Numpy_INCLUDE_DIRS numpy/arrayobject.h HINTS ${Numpy_PREFIX})
#    IF (Numpy_INCLUDE_DIRS)
#      MESSAGE(STATUS "Numpy_INCLUDE_DIRS: ${Numpy_INCLUDE_DIRS}")
#    ELSE()
#      MESSAGE(FATAL_ERROR "Numpy headers not found, but needed when CompadreHarness_USE_PYTHON:BOOL=ON. Try setting Numpy_PREFIX.")
#    ENDIF()
#
#    set(SWIG_PREFIX "../GMLS/src/swig") # relative to examples folder
#    if (PYTHON_VERSION_MAJOR EQUAL 3)
#      set(GMLS_Module_DEST "lib/python3/dist-packages")
#    else()
#      set(GMLS_Module_DEST "lib/python/dist-packages")
#    endif()
#  ENDIF()
#
#endif()


##########
#
#
#  OPTIONS
#
#
##########


#bob_option(Toolkit_TESTS "Compile GMLS tests" ON)
#bob_option(Toolkit_EXAMPLES "Compile GMLS examples" "${Toolkit_TESTS}")

bob_option(CompadreHarness_TESTS "Compile CompadreHarness tests" ON)
bob_option(CompadreHarness_EXAMPLES "Compile CompadreHarness examples" "${CompadreHarness_TESTS}")

# physics compiled by default if CompadreHarness_EXAMPLES is ON and not if OFF
bob_option(CompadreHarness_PHYSICS "Compile CompadreHarness Physics" "${CompadreHarness_EXAMPLES}")


bob_option(CompadreHarness_CANGA_ENABLED "Compile optional CANGA tests and examples" OFF)

##########
#
#
#  Kokkos Configurations
#
#
##########


##  Detect KokkosCore functionality from Trilinos
include(${CMAKE_CURRENT_LIST_DIR}/cmake/detect_trilinos_opts.cmake)
detect_trilinos_opts()
if (KokkosCore_HAS_CUDA)
  if (NOT KokkosCore_HAS_CUDA_LAMBDA)
    message(FATAL_ERROR
      "Please reconfigure Kokkos with -DKokkos_ENABLE_Cuda_Lambda:BOOL=ON")
  else()
    bob_option(CompadreHarness_USE_CUDA "Whether to use CUDA" ON)
  endif()
endif()


###
###   OpenMP / Cuda
###
#if (KokkosCore_HAS_CUDA)
#  find_package(CUDA)
#  bob_input(CompadreHarness_CUDA_ARCH "" STRING "CUDA arch specifier")
#  bob_option(CompadreHarness_USE_CUDA "Whether to use CUDA" ON)
#endif()
#bob_option(Compadre_USE_OPENMP "Whether to use OpenMP" "${KokkosCore_HAS_OpenMP}")



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
  #if(CompadreHarness_USE_CUDA)
  #  if (CompadreHarness_CUDA_ARCH)
  #    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:--cuda-gpu-arch=${CompadreHarness_CUDA_ARCH}>)
  #  endif()
  #  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-x>)
  #  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:cuda>)
  #  # -Wweak-vtables gives false positives with -x cuda
  #  set(FLAGS "${FLAGS} -Wno-weak-vtables")
  #endif()
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    #if(CompadreHarness_USE_CUDA)
    #ifset(FLAGS "${FLAGS} -expt-extended-lambda -lineinfo")
    #ifif (CompadreHarness_CUDA_ARCH)
    #if  set(FLAGS "${FLAGS} -arch=${CompadreHarness_CUDA_ARCH}")
    #ifendif()
    #ifif (CUDA_VERSION EQUAL 9)
    #if  set(FLAGS "${FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    #ifendif()
    #else()
    set(FLAGS "${FLAGS} -fno-omit-frame-pointer")
    if(Compadre_USE_OPENMP)
      set(FLAGS "${FLAGS} -fopenmp")
    endif()
    #endif()
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  set(FLAGS "${FLAGS} -fno-omit-frame-pointer")
  if(Compadre_USE_OPENMP)
    set(FLAGS "${FLAGS} -fopenmp")
  endif()
else()
  message(WARNING "Unexpected compiler type ${CMAKE_CXX_COMPILER_ID}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")


if (CompadreHarness_USE_Trilinos_CXX_Flags)
  # Get compiler flags from Trilinos
  SET(CMAKE_CXX_FLAGS "")
  IF (Trilinos_CXX_COMPILER_FLAGS) 
    FOREACH(arg ${Trilinos_CXX_COMPILER_FLAGS} )
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${arg}")
    ENDFOREACH()
  ENDIF()
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  # if an Intel compiler, skip all library searching and add "-mkl" to flags
  SET(CMAKE_CXX_FLAGS "-mkl ${CMAKE_CXX_FLAGS}")
endif()

bob_end_cxx_flags()




##########
#
#
#  PREPROCESSOR DEFINES FOR USE WHEN BUILDING CODE
#
#
##########


set(CompadreHarness_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

set(CompadreHarness_KEY_BOOLS
    CompadreHarness_USE_LAPACK
    CompadreHarness_DEBUG
    CompadreHarness_USE_Compose
    CompadreHarness_USE_OPENBLAS
    CompadreHarness_USE_Nanoflann
    CompadreHarness_USE_Netcdf
    CompadreHarness_USE_Netcdf_MPI
    CompadreHarness_USE_PYTHON
    TRILINOS_LINEAR_SOLVES
    LAPACK_DECLARED_THREADSAFE
   )

set(CompadreHarness_KEY_INTS
    CompadreHarness_VERSION_MAJOR
    CompadreHarness_VERSION_MINOR
    CompadreHarness_VERSION_PATCH
   )

set(CompadreHarness_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CompadreHarness_KEY_STRINGS
    CompadreHarness_SEMVER
    CompadreHarness_COMMIT
    CompadreHarness_CXX_FLAGS
    CompadreHarness_CMAKE_ARGS
    GMLS_Module_DEST
    CompadreHarness_INSTALL_PREFIX
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

# check that either CUDA or LAPACK were turned on (needed for any computation to be performed)

bob_option(CompadreHarness_DEBUG "Debug Mode" ON)
bob_option(Compadre_DEBUG "Debug Mode" "${CompadreHarness_DEBUG}")
option(Compadre_USE_Trilinos "" ON)
add_subdirectory(toolkit)

#set(Toolkit_PREFIX "${PROJECT_BINARY_DIR}/Toolkit-install")
#set(Toolkit_ARGS
#  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
#  "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
#  "-DCMAKE_INSTALL_PREFIX=${Toolkit_PREFIX}"
#  "-DCompadre_USE_CUDA:BOOL=${CompadreHarness_USE_CUDA}"
#  "-DCompadre_USE_LAPACK:BOOL=${CompadreHarness_USE_LAPACK}"
#  "-DLAPACK_DECLARED_THREADSAFE:BOOL=${LAPACK_DECLARED_THREADSAFE}"
#  "-DCompadre_USE_PYTHON:BOOL=${CompadreHarness_USE_PYTHON}"
#  "-DCompadre_EXAMPLES:BOOL=${Toolkit_EXAMPLES}"
#  "-DTrilinos_PREFIX=${Trilinos_PREFIX}"
#  "-DCompadre_USE_Trilinos:BOOL=ON"
#  "-DCompadre_TESTS:BOOL=${Toolkit_TESTS}"
#  "-DCompadre_DEBUG:BOOL=${Toolkit_DEBUG}"
#)

    #GIT_REPOSITORY "/ascldap/users/pakuber/Desktop/ProjectStubs/Particles/barebone_toolkit"

# RPATH related settings
# https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#SET(CMAKE_INSTALL_RPATH ${Toolkit_PREFIX}/lib ${CMAKE_INSTALL_RPATH})
#SET(CMAKE_BUILD_RPATH ${Toolkit_PREFIX}/lib ${CMAKE_BUILD_RPATH})
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_RPATH})

#externalproject_add(Toolkit
#    PREFIX toolkit
#    SOURCE_DIR "${PROJECT_SOURCE_DIR}/toolkit"
#    CMAKE_ARGS ${Toolkit_ARGS}
#    BUILD_ALWAYS 1
#    TEST_AFTER_INSTALL OFF
#    )

#    GIT_REPOSITORY "/ascldap/users/pakuber/Desktop/ProjectStubs/Particles/barebone_toolkit"
# UPDATE_DISCONNECTED 1 indicates that GIT_TAG is a hash and it is stable
#SOURCE_DIR "${PROJECT_SOURCE_DIR}/toolkit"
#    UPDATE_COMMAND "" 
#DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/toolkit"
#    GIT_REPOSITORY "git+ssh://software.sandia.gov/git/compadre-dev"
#    GIT_TAG "e590e1bba62823"

bob_option(CompadreHarness_USE_Compose "Build compose for shape preservation and conservation post-processing" OFF)
if(CompadreHarness_USE_Compose)

  ##########
  #
  #
  #  COMPOSE LIBRARY (ONLY NEEDED IF HARNESS IS BEING BUILT)
  #
  #
  ##########
  
  set (Compose_Kokkos_DIR "${Trilinos_PREFIX}/lib/cmake/Kokkos")
  set(Compose_PREFIX "${PROJECT_BINARY_DIR}/Compose-install")
  set(Compose_ARGS
    "-DCMAKE_INSTALL_PREFIX=${Compose_PREFIX}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DKokkos_DIR=${Compose_Kokkos_DIR}"
    "-DKOKKOS_IN_TRILINOS=ON"
  )
  
  # RPATH related settings
  # https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
  SET(CMAKE_INSTALL_RPATH ${Compose_PREFIX}/lib ${CMAKE_INSTALL_RPATH})
  SET(CMAKE_BUILD_RPATH ${Compose_PREFIX}/lib ${CMAKE_BUILD_RPATH})
  
  externalproject_add(Compose
      PREFIX compose
      SOURCE_DIR "${PROJECT_SOURCE_DIR}/compose"
      CMAKE_ARGS ${Compose_ARGS}
      BUILD_ALWAYS 1
      TEST_AFTER_INSTALL OFF
  )
  add_definitions (-DCOMPOSE_CONFIG_IS_CMAKE)
endif()

add_subdirectory(src)

if(CompadreHarness_PHYSICS)
  add_subdirectory(physics)
endif()

if(CompadreHarness_EXAMPLES)
  add_subdirectory(examples)
endif()



##########
#
#
#  DOXYGEN
#
#
##########
bob_option(CompadreHarness_BUILD_Doxygen "Compile Doxygen documentation" ON)
if(CompadreHarness_BUILD_Doxygen)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in" "${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile"
                 @ONLY)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/DoxygenLayout.xml" "${CMAKE_CURRENT_BINARY_DIR}/doc/DoxygenLayout.xml"
                 COPYONLY)
  add_custom_target(HarnessDoxygen
                    COMMAND doxygen
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc )
  # still must be called manually with "make Doxygen"
endif(CompadreHarness_BUILD_Doxygen)


set (augmented_list "${Trilinos_LIBRARIES}")
bob_end_package_no_recurse("${augmented_list}")
