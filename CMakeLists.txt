cmake_minimum_required(VERSION 3.0.0)

##########
#
#   Compadre Details
#
########## 
PROJECT(Compadre VERSION 1.0.0 LANGUAGES CXX)

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



#LAPACK
bob_option(Compadre_USE_LAPACK "Use LAPACK" ON)
SET(Compadre_USE_OPENBLAS OFF) # need to know if openblas is used in order to limit its ability to spawn threads
bob_input(LAPACK_PREFIX "" PATH "Path to LAPACK install")
if (LAPACK_PREFIX)
  set(CMAKE_PREFIX_PATH ${LAPACK_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
bob_input(LAPACK_LOCATION "" PATH "Optional explicit path to exact location of LAPACK")
bob_option(LAPACK_DECLARED_THREADSAFE "Is LAPACK threadsafe? If not, default to serial calls for solves." ON)
if (Compadre_USE_LAPACK)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    # Intel compiler detected. Will use "-mkl" and no need to find Lapack or Blas
  else()
    if ("${LAPACK_LOCATION}" STREQUAL "")
      FIND_LIBRARY(LAPACK_LIB lapack)
    else()
      SET(LAPACK_LIB "${LAPACK_LOCATION}")
    endif()
    MESSAGE(STATUS "LAPACK_LIB: ${LAPACK_LIB}")
    IF (NOT(LAPACK_LIB))
      MESSAGE(FATAL_ERROR "Compadre_USE_LAPACK is ON, but LAPACK_PREFIX was set incorrectly.")
    ELSE()
      IF(APPLE)
        execute_process(COMMAND otool -L  "${LAPACK_LIB}" OUTPUT_VARIABLE LAPACK_DEPENDENCIES) 
      ELSE()
        execute_process(COMMAND ldd "${LAPACK_LIB}" OUTPUT_VARIABLE LAPACK_DEPENDENCIES) 
      ENDIF()
      string(FIND "${LAPACK_DEPENDENCIES}" "openblas" BLAS_MATCH)
      if ("${BLAS_MATCH}" GREATER "-1")
        SET(Compadre_USE_OPENBLAS ON)
      endif()
    ENDIF()
  endif()
endif()

#TRILINOS
bob_option(Compadre_USE_Trilinos "Use Trilinos" OFF)
bob_input(Trilinos_PREFIX "" PATH "Path to Trilinos install")
if (Trilinos_PREFIX)
  set(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
IF (Compadre_USE_Trilinos)
  FIND_PACKAGE(Trilinos)
  set(Compadre_USE_Trilinos_CXX_Flags ON)
  bob_add_dependency(PUBLIC NAME Trilinos
    TARGETS "${Trilinos_LIBRARIES}"
    INCLUDE_DIR_VARS 
      Trilinos_INCLUDE_DIRS 
      Trilinos_TPL_INCLUDE_DIRS 
    LIBRARY_VARS
  Trilinos_LIBRARIES Trilinos_TPL_LIBRARIES)

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
else()
  MESSAGE(FATAL_ERROR "Compadre_USE_KokkosCore set to OFF, but entire toolkit relies on KokkosCore.")
endif()

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

    set(SWIG_PREFIX "../python") # relative to examples folder
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



bob_option(Compadre_TESTS "Compile Compadre tests" ON)
bob_option(Compadre_EXAMPLES "Compile Compadre examples" "${Compadre_TESTS}")



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
  if (KokkosCore_HAS_CUDA)
    if (NOT KokkosCore_HAS_CUDA_LAMBDA)
      message(FATAL_ERROR
        "Please reconfigure Kokkos with -DKokkos_ENABLE_Cuda_Lambda:BOOL=ON")
    else()
      bob_option(Compadre_USE_CUDA "Whether to use CUDA" ON)
    endif()
  endif()
ELSE()
  if (KOKKOS_INTERNAL_USE_CUDA EQUAL "1")
    # Find Cuda using hints from kokkos generate makefile
    set(CMAKE_PREFIX_PATH "${KOKKOS_GMAKE_TPL_LIBRARY_DIRS}/.." ${CMAKE_PREFIX_PATH})
    find_path(CUDA_INCLUDE_DIRS NAMES "cublas_api.h" HINT "${KOKKOS_GMAKE_TPL_LIBRARY_DIRS}/..")
    MESSAGE(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    set(CMAKE_PREFIX_PATH ${KOKKOS_GMAKE_TPL_LIBRARY_DIRS} ${CMAKE_PREFIX_PATH})
    find_library(CUDA_CUBLAS_LIBRARIES cublas HINT "${CUDA_INCLUDE_DIRS}/../lib64" NO_DEFAULT_PATH)
    find_library(CUDA_CUSOLVER_LIBRARIES cusolver HINT "${CUDA_INCLUDE_DIRS}/../lib64" NO_DEFAULT_PATH)
    MESSAGE(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS}/../lib64")
    bob_option(Compadre_USE_CUDA "Whether to use CUDA" ON)
  endif()
ENDIF()


##
##   OpenMP / Cuda
##
if (KokkosCore_HAS_CUDA)
  find_package(CUDA)
  bob_input(Compadre_CUDA_ARCH "" STRING "CUDA arch specifier")
  bob_option(Compadre_USE_CUDA "Whether to use CUDA" ON)
endif()
bob_option(Compadre_USE_OpenMP "Whether to use OpenMP" "${KokkosCore_HAS_OpenMP}")



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


set(Compadre_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

set(Compadre_KEY_BOOLS
    Compadre_USE_CUDA
    Compadre_USE_KokkosCore
    Compadre_USE_LAPACK
    Compadre_USE_OPENBLAS
    Compadre_USE_MPI
    Compadre_USE_OpenMP
    Compadre_USE_PYTHON
    Compadre_USE_Trilinos
    LAPACK_DECLARED_THREADSAFE
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
#  COMPADRE LIBRARIES
#
#
##########

# check that either CUDA or LAPACK were turned on (needed for any computation to be performed)

if (Compadre_USE_CUDA OR Compadre_USE_LAPACK)

  add_subdirectory(src)
  
  if(Compadre_EXAMPLES)
    add_subdirectory(examples)
  endif()

  if(Compadre_USE_PYTHON)
    add_subdirectory(python)
  endif()

else ()
  MESSAGE(FATAL_ERROR "Both Compadre_USE_CUDA and Compadre_USE_LAPACK are currently OFF.\n Since these are both OFF, either Compadre_USE_LAPACK is set to OFF explicitly in the build script, or CUDA was not found in the KokkosCore specified (Compadre_USE_CUDA is a calculated variable, not an option to be set).")
endif () 



##########
#
#
#  DOXYGEN
#
#
##########
bob_option(Compadre_BUILD_Doxygen "Compile Doxygen documentation" ON)
if(Compadre_BUILD_Doxygen)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in" "${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile"
                 @ONLY)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/DoxygenLayout.xml" "${CMAKE_CURRENT_BINARY_DIR}/doc/DoxygenLayout.xml"
                 COPYONLY)
  add_custom_target(Doxygen
                    COMMAND doxygen
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc )
  # still must be called manually with "make Doxygen"
endif(Compadre_BUILD_Doxygen)


set (augmented_list "${Trilinos_LIBRARIES}" "${VTK_LIBRARIES}")
bob_end_package_no_recurse("${augmented_list}")
