cmake_minimum_required(VERSION 3.0.0)

##########
#
#   Compadre Details
#
########## 

PROJECT(Compadre VERSION 1.0.2 LANGUAGES CXX)

# cmake helper functions
include(${CMAKE_CURRENT_LIST_DIR}/cmake/bob.cmake)

bob_begin_package()

# Set to OFF for significantly faster performance and ON for error tracking
bob_option(Compadre_DEBUG "Run Compadre Toolkit in DEBUG mode" ON)
bob_option(Compadre_EXTREME_DEBUG "Run Compadre Toolkit in EXTREME DEBUG mode" OFF)

# RPATH related settings
# https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_RPATH})

# RPATH should always include the folder it is called from
bob_option(PYTHON_CALLING_BUILD "Python setuptools calling build" OFF)
bob_option(Compadre_USE_PYTHON "Use PYTHON" OFF)
if (Compadre_USE_PYTHON)
  if (APPLE)
    SET(CMAKE_BUILD_RPATH ${CMAKE_BUILD_RPATH} "@loader_path/")
    SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} "@loader_path/")
    if (NOT(PYTHON_CALLING_BUILD))
      # If python is calling the build, it will provide these flags
      SET(CMAKE_PYTHON_SHARED_LINKER_FLAGS "-undefined dynamic_lookup -nodefaultlibs")
    endif()
  else()
    SET(CMAKE_BUILD_RPATH ${CMAKE_BUILD_RPATH} "$ORIGIN/")
    SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} "$ORIGIN/")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,origin")
  endif()
endif()


include(ExternalProject)


##########
#
#
#  TPLS
#
#
##########



#TRILINOS
bob_input(Trilinos_PREFIX "" PATH "Path to Trilinos install")
if (Trilinos_PREFIX)
  set(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})
  set(Compadre_USE_Trilinos ON)
endif()

#KOKKOS
bob_option(Compadre_USE_KokkosCore "Use KokkosCore" ON)
bob_input(KokkosCore_PREFIX "${Trilinos_PREFIX}" PATH "Path to KokkosCore install")

#KOKKOS-KERNELS
bob_option(Compadre_USE_KokkosKernels "Use KokkosKernels" ON)
bob_input(KokkosKernels_PREFIX "${Trilinos_PREFIX}" PATH "Path to KokkosKernels install")

if (Trilinos_PREFIX AND NOT("${KokkosCore_PREFIX}" STREQUAL "${Trilinos_PREFIX}"))
    # check both KokkosCore_PREFIX and Trilinos_PREFIX were specified differ (if either or both are not specified, it is fine)
    MESSAGE(FATAL_ERROR "KokkosCore_PREFIX and Trilinos_PREFIX were both specified, but with different values")
endif()

if (APPLE)
  bob_option(Compadre_USE_CUDA "Whether to use CUDA" OFF)
  bob_option(Compadre_USE_OPENMP "Whether to use OPENMP" OFF)
  bob_option(Compadre_USE_PTHREAD "Whether to use PTHREAD" ON)
else()
  bob_option(Compadre_USE_CUDA "Whether to use CUDA" OFF)
  bob_option(Compadre_USE_OPENMP "Whether to use OPENMP" ON)
  bob_option(Compadre_USE_PTHREAD "Whether to use PTHREAD" OFF)
endif()
bob_option(Kokkos_ENABLE_PTHEAD "Whether to use PTHREAD when building KokkosCore" "${Compadre_USE_PTHREAD}")
bob_option(Kokkos_ENABLE_OPENMP "Whether to use OPENMP when building KokkosCore" "${Compadre_USE_OPENMP}")
bob_option(Kokkos_ENABLE_CUDA "Whether to use CUDA when building KokkosCore" "${Compadre_USE_CUDA}")
bob_option(Compadre_EXPLICITLY_DISABLE_CUDA "Special case CUDA override" OFF)

if (Compadre_USE_OPENMP AND NOT(Compadre_USE_CUDA))
  # If cuda is enabled, nvcc_wrapper will take of openmp flag
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set(KOKKOS_IN_TRILINOS OFF)
set(KOKKOS_EXISTING_ELSEWHERE OFF)
set(KOKKOS_BUILT_FOR_USER OFF)
set(KOKKOSKERNELS_IN_TRILINOS OFF)
set(KOKKOSKERNELS_EXISTING_ELSEWHERE OFF)
set(KOKKOSKERNELS_BUILT_FOR_USER OFF)

#LAPACK and BLAS
set(Compadre_USE_LAPACK ON)
if (Compadre_USE_CUDA)
  set(Compadre_USE_LAPACK OFF)
  message(STATUS "Compadre_USE_LAPACK set to OFF because Compadre_USE_CUDA is ON")
endif()
message(STATUS "Compadre_USE_LAPACK: ${Compadre_USE_LAPACK}")
bob_option(LAPACK_DECLARED_THREADSAFE "Is LAPACK threadsafe? If not, default to serial calls for solves." ON)

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

  LIST(FIND Trilinos_PACKAGE_LIST KokkosCore KokkosCoreID)
  IF (KokkosCoreID GREATER -1 )
    MESSAGE(STATUS "Found KokkosCore inside Trilinos!")
    set(KOKKOS_IN_TRILINOS ON)
    set(KokkosCore_FOUND ON)
  ELSE()
    MESSAGE(FATAL_ERROR "Found Trilinos but could not find KokkosCore.")
  ENDIF()

  LIST(FIND Trilinos_PACKAGE_LIST KokkosKernels KokkosKernelsID)
  IF (KokkosKernelsID GREATER -1 )
    MESSAGE(STATUS "Found KokkosKernels inside Trilinos!")
    set(KOKKOSKERNELS_IN_TRILINOS ON)
    set(KokkosKernels_FOUND ON)
  ELSE()
    MESSAGE(FATAL_ERROR "Found Trilinos but could not find KokkosKernels.")
  ENDIF()

  if(KOKKOSKERNELS_IN_TRILINOS AND NOT(KOKKOS_IN_TRILINOS))
      MESSAGE(FATAL_ERROR "KokkosKernels found in Trilinos, but not Kokkos")
  elseif(KOKKOS_IN_TRILINOS AND NOT(KOKKOSKERNELS_IN_TRILINOS))
      MESSAGE(FATAL_ERROR "Kokkos found in Trilinos, but not KokkosKernels")
  endif()

  #  Detect KokkosCore functionality from Trilinos
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/detect_trilinos_opts.cmake)
  detect_trilinos_opts()

  # get kokkos settings, change our settings, error to user if we something they requested is not enabled
  if (KokkosCore_HAS_CUDA AND NOT(Compadre_EXPLICITLY_DISABLE_CUDA))
    if (NOT KokkosCore_HAS_CUDA_LAMBDA)
      message(FATAL_ERROR "Please reconfigure Trilinos with -DKokkos_ENABLE_Cuda_Lambda:BOOL=ON or set Compadre_EXPLICITLY_DISABLE_CUDA to ON")
    endif()
    message(STATUS "CUDA enabled in KokkosCore in Trilinos, setting Compadre_USE_CUDA to ON")
    set(Compadre_USE_CUDA ON)
  else()
    if (Compadre_USE_CUDA)
      message(FATAL_ERROR "Compadre_USE_CUDA is ON, but the Kokkos at Trilinos_PREFIX does not have CUDA enabled.")
    endif()
  endif()

  if (KokkosCore_HAS_OPENMP)
    message(STATUS "OPENMP enabled in KokkosCore in Trilinos, setting Compadre_USE_OPENMP to ON")
    set(Compadre_USE_OPENMP ON)
  else()
    if (Compadre_USE_OPENMP)
      message(FATAL_ERROR "Compadre_USE_OPENMP is ON, but the Kokkos at Trilinos_PREFIX does not have OPENMP enabled.")
    endif()
  endif()

  if (KokkosCore_HAS_PTHREAD)
    message(STATUS "PTHREAD enabled in KokkosCore in Trilinos, setting Compadre_USE_PTHREAD to ON")
    set(Compadre_USE_PTHREAD ON)
  else()
    if (Compadre_USE_PTHREAD)
      message(FATAL_ERROR "Compadre_USE_PTHREAD is ON, but the Kokkos at Trilinos_PREFIX does not have PTHREAD enabled.")
    endif()
  endif()

ENDIF()

if (KOKKOS_IN_TRILINOS)
  bob_add_dependency(PUBLIC NAME KokkosCore TARGETS kokkoscore INCLUDE_DIR_VARS KokkosCore_INCLUDE_DIRS)
else()
  if(NOT(Compadre_USE_KokkosCore))
    MESSAGE(FATAL_ERROR "Compadre_USE_KokkosCore set to OFF, but entire toolkit relies on KokkosCore.")
  else()
    if (KokkosCore_PREFIX STREQUAL "")
      SET(KOKKOS_BUILT_FOR_USER ON)
      option(Kokkos_ENABLE_OPENMP "" "${Compadre_USE_OPENMP}")
      option(Kokkos_ENABLE_PTHREAD  "" "${Compadre_USE_PTHREAD}")
      option(Kokkos_ENABLE_CUDA "" "${Compadre_USE_CUDA}")
      option(Kokkos_ENABLE_CUDA_LAMBDA "" "${Compadre_USE_CUDA}")
      option(Kokkos_ENABLE_TESTS "" OFF)
      option(Kokkos_ENABLE_EXAMPLES "" OFF)
      add_subdirectory(kokkos)
      if (NOT(KokkosKernels_PREFIX STREQUAL ""))
          MESSAGE(FATAL_ERROR "KokkosKernels_PREFIX specified but KokkosCore_PREFIX not specified. \
          Either provide KokkosCore_PREFIX as well, or remove KokkosKernels_PREFIX specification.")
      endif()
    else()
      # Existing KokkosCore indicated by user by setting KokkosCore_PREFIX, so we look for it
      # where specified (only)
      set(CMAKE_PREFIX_PATH ${KokkosCore_PREFIX} ${CMAKE_PREFIX_PATH})
      find_package(Kokkos PATHS "${KokkosCore_PREFIX}" NO_DEFAULT_PATH REQUIRED)
      set(KOKKOS_EXISTING_ELSEWHERE ON)
    endif()
  endif()
endif()

if (KOKKOSKERNELS_IN_TRILINOS)
  bob_add_dependency(PUBLIC NAME KokkosKernels TARGETS kokkoskernels INCLUDE_DIR_VARS KokkosKernels_INCLUDE_DIRS)
  if (Compadre_USE_LAPACK)
    list(FIND Trilinos_TPL_LIST "LAPACK" LAPACKID)
    if (NOT(LAPACKID GREATER -1))
      MESSAGE(FATAL_ERROR "Compadre_USE_LAPACK set to ON, but LAPACK not defined in Trilinos")
    endif()
    list(FIND Trilinos_TPL_LIST "BLAS" BLASID)
    if (NOT(BLASID GREATER -1))
        MESSAGE(FATAL_ERROR "Compadre_USE_LAPACK set to ON, but BLAS not defined in Trilinos")
    endif()
  endif()
  if (Compadre_US_CUDA)
    list(FIND Trilinos_TPL_LIST "CUDA" CUDAID)
    if (NOT(CUDAID GREATER -1))
        MESSAGE(FATAL_ERROR "Compadre_USE_CUDA set to ON, but CUDA not defined in Trilinos")
    endif()
  endif()
else()
  if(NOT(Compadre_USE_KokkosKernels))
    MESSAGE(FATAL_ERROR "Compadre_USE_KokkosKernels set to OFF, but entire toolkit relies on KokkosKernels.")
  else()
    if (KokkosKernels_PREFIX STREQUAL "")
      SET(KOKKOSKERNELS_BUILT_FOR_USER ON)
      # both are required to be sure it finds it
      OPTION(KokkosKernels_ENABLE_TPL_LAPACK "" "${Compadre_USE_LAPACK}")
      SET(KokkosKernels_ENABLE_TPL_LAPACK "${Compadre_USE_LAPACK}")
      OPTION(KokkosKernels_ENABLE_TPL_BLAS "" "${Compadre_USE_LAPACK}")
      SET(KokkosKernels_ENABLE_TPL_BLAS "${Compadre_USE_LAPACK}")
      OPTION(KokkosKernels_ENABLE_TPL_CUSPARSE "" "${Compadre_USE_CUDA}")
      SET(KokkosKernels_ENABLE_TPL_CUSPARSE "${Compadre_USE_CUDA}")
      OPTION(KokkosKernels_ENABLE_TPL_CUBLAS "" "${Compadre_USE_CUDA}")
      SET(KokkosKernels_ENABLE_TPL_CUBLAS "${Compadre_USE_CUDA}")
      SET(KokkosKernels_ENABLE_EXAMPLES OFF)
      SET(KokkosKernels_ENABLE_TESTS OFF)
      OPTION(KokkosKernels_ETI_ONLY "" OFF)
      SET(KokkosKernels_ETI_ONLY OFF)
      OPTION(KokkosKernels_INST_OFFSET_SIZE_T "" OFF)
      SET(KokkosKernels_INST_OFFSET_SIZE_T OFF)
      OPTION(KokkosKernels_INST_OFFSET_INT "" OFF)
      SET(KokkosKernels_INST_OFFSET_INT OFF)
      OPTION(KokkosKernels_INST_LAYOUTLEFT "" OFF)
      SET(KokkosKernels_INST_LAYOUTLEFT OFF)
      OPTION(KokkosKernels_INST_LAYOUTRIGHT "" OFF)
      SET(KokkosKernels_INST_LAYOUTRIGHT OFF)
      OPTION(KokkosKernels_INST_DOUBLE "" OFF)
      SET(KokkosKernels_INST_DOUBLE OFF)
      OPTION(KokkosKernels_INST_ORDINAL_INT "" OFF)
      SET(KokkosKernels_INST_ORDINAL_INT OFF)
      add_subdirectory(kokkos-kernels)
      if (NOT(KokkosCore_PREFIX STREQUAL ""))
          MESSAGE(FATAL_ERROR "KokkosCore_PREFIX specified but KokkosKernels_PREFIX not specified. \
          Either provide KokkosKernels_PREFIX as well, or remove KokkosCore_PREFIX specification.")
      endif()
    else()
      # Existing KokkosCore indicated by user by setting KokkosCore_PREFIX, so we look for it
      # where specified (only)
      set(CMAKE_PREFIX_PATH ${KokkosKernels_PREFIX} ${CMAKE_PREFIX_PATH})
      find_package(KokkosKernels PATHS "${KokkosKernels_PREFIX}" NO_DEFAULT_PATH REQUIRED)
      if(Compadre_USE_LAPACK AND NOT(TARGET Kokkos::LAPACK))
        MESSAGE(FATAL_ERROR "Compadre_USE_LAPACK set to ON, but Kokkos::LAPACK not defined")
      endif()
      if(Compadre_USE_BLAS AND NOT(Kokkos::BLAS))
        MESSAGE(FATAL_ERROR "Compadre_USE_BLAS set to ON, but Kokkos::BLAS not defined")
      endif()
      if(Compadre_USE_CUDA AND NOT(Kokkos::CUDA))
          MESSAGE(FATAL_ERROR "Compadre_USE_CUDA set to ON, but Kokkos::CUDA not defined")
      endif()
      set(KOKKOSKERNELS_EXISTING_ELSEWHERE ON)
    endif()
  endif()
endif()

#MPI
FIND_PACKAGE(MPI QUIET)
bob_option(Compadre_USE_MPI "Use MPI for parallelism" ${MPI_CXX_FOUND})
message(STATUS "MPI Enabled: ${MPI_CXX_FOUND}")
if (MPI_CXX_FOUND)
    MESSAGE(STATUS "MPI_CXX_INCLUDE_PATH: ${MPI_CXX_INCLUDE_PATH}")
    MESSAGE(STATUS "MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")
endif()



#PYTHON
bob_option(Compadre_USE_MATLAB "Use MATLAB interface for PYTHON" OFF)
bob_input(PYTHON_PREFIX "" PATH "Path to PYTHON install")
if (PYTHON_PREFIX)
  set(CMAKE_PREFIX_PATH ${PYTHON_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
if (Compadre_USE_PYTHON)

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

  IF(NOT EXISTS ${Numpy_PREFIX})
    EXECUTE_PROCESS(
      COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
      OUTPUT_VARIABLE Numpy_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE )
  ENDIF()
  MESSAGE(STATUS "Numpy_PREFIX: ${Numpy_PREFIX}")
  
  FIND_PATH(Numpy_INCLUDE_DIRS numpy/arrayobject.h HINTS ${Numpy_PREFIX})
  IF (Numpy_INCLUDE_DIRS)
    MESSAGE(STATUS "Numpy_INCLUDE_DIRS: ${Numpy_INCLUDE_DIRS}")
  ELSE()
    MESSAGE(FATAL_ERROR "Numpy headers not found, but needed when Compadre_USE_PYTHON:BOOL=ON. Try setting Numpy_PREFIX.")
  ENDIF()

  set(SWIG_PREFIX "../python") # relative to examples folder
  bob_input(GMLS_Module_DEST "lib/python/dist-packages" PATH "Path to install python module")

  IF(NOT EXISTS ${PYTHON_INCLUDE_DIRS})
    EXECUTE_PROCESS(
      COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
      OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
      OUTPUT_STRIP_TRAILING_WHITESPACE )
  ENDIF()

  EXECUTE_PROCESS(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))"
    OUTPUT_VARIABLE PYTHON_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE )
  MESSAGE(STATUS "PYTHON_PATHS: ${PYTHON_PATHS}")

  file(GLOB PYTHON_LIBNAME
    "${PYTHON_PATHS}/libpython*${CMAKE_SHARED_LIBRARY_SUFFIX}"
  )
  get_filename_component(PYTHON_LIBNAME_BASE "${PYTHON_LIBNAME}" NAME)
  STRING(REGEX REPLACE "lib(.*)${CMAKE_SHARED_LIBRARY_SUFFIX}" "\\1" PYTHON_LIBNAME "${PYTHON_LIBNAME_BASE}")

  MESSAGE(STATUS "PYTHON_LIBNAME: ${PYTHON_LIBNAME}")
  find_library(PYTHON_LIBRARIES NAME "${PYTHON_LIBNAME}" PATHS "${PYTHON_PATHS}" NO_DEFAULT_PATH)

  MESSAGE(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
  MESSAGE(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")

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
#  CXX FLAGS
#
#
##########




if(KOKKOS_IN_TRILINOS)
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
      if(Compadre_USE_OPENMP)
        set(FLAGS "${FLAGS} -fopenmp")
      endif()
    endif()
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(FLAGS "${FLAGS} -fno-omit-frame-pointer")
    if(Compadre_USE_OPENMP)
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
    Compadre_DEBUG
    Compadre_EXTREME_DEBUG
    Compadre_USE_CUDA
    Compadre_USE_KokkosCore
    Compadre_USE_LAPACK
    Compadre_USE_OPENBLAS
    Compadre_USE_MPI
    Compadre_USE_OPENMP
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


if (NOT(PYTHON_CALLING_BUILD))
  bob_get_commit()
  bob_get_semver()
endif()



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
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in" "${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile"
                 @ONLY)
  add_custom_target(Doxygen
                    COMMAND doxygen
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc )
  # still must be called manually with "make Doxygen"
endif(Compadre_BUILD_Doxygen)


set (augmented_list "${Trilinos_LIBRARIES}" "${VTK_LIBRARIES}")
bob_end_package_no_recurse("${augmented_list}")
