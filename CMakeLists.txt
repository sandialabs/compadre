cmake_minimum_required(VERSION 3.0.0)

##########
#
#   CompadreHarness Details
#
########## 
PROJECT(CompadreHarness VERSION 1.0.0 LANGUAGES CXX)

# cmake helper functions
include(${CMAKE_CURRENT_LIST_DIR}/toolkit/cmake/bob.cmake)

bob_begin_package()

include(ExternalProject)
bob_input(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib" PATH "")

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

LIST(REVERSE Trilinos_INCLUDE_DIRS)
LIST(REMOVE_DUPLICATES Trilinos_INCLUDE_DIRS)
LIST(REVERSE Trilinos_INCLUDE_DIRS)

LIST(REVERSE Trilinos_LIBRARIES)
LIST(REMOVE_DUPLICATES Trilinos_LIBRARIES)
LIST(REVERSE Trilinos_LIBRARIES)

LIST(REVERSE Trilinos_TPL_LIBRARIES)
LIST(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
LIST(REVERSE Trilinos_TPL_LIBRARIES)

MESSAGE("\nFound Trilinos for harness!  Here are the details: ")
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
LIST(FIND Trilinos_PACKAGE_LIST Intrepid IntrepidTrilinosID)
LIST(FIND Trilinos_PACKAGE_LIST Shards ShardsTrilinosID)

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

SET(TRILINOS_DISCRETIZATION OFF)
IF (IntrepidTrilinosID GREATER -1 )
  MESSAGE(STATUS "Found Intrepid inside Trilinos!")
ELSE()
  MESSAGE(STATUS "Found Trilinos but could not find Intrepid.")
ENDIF()
IF (ShardsTrilinosID GREATER -1 )
  MESSAGE(STATUS "Found Shards inside Trilinos!")
ELSE()
  MESSAGE(STATUS "Found Trilinos but could not find Shards.")
ENDIF()
IF ((IntrepidTrilinosID GREATER -1 ) AND (ShardsTrilinosID GREATER -1 ))
  MESSAGE(STATUS "All packages needed for discretization in Trilinos found.")
  SET(TRILINOS_DISCRETIZATION ON)
ENDIF()



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
   


##########
#
#
#  OPTIONS
#
#
##########


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


# All Kokkos configuration is handles in the ./toolkit subdirectory (flags pass seamlessly between the two projects)


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
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
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
    CompadreHarness_DEBUG
    CompadreHarness_USE_Compose
    CompadreHarness_USE_Netcdf
    CompadreHarness_USE_Netcdf_MPI
    TRILINOS_LINEAR_SOLVES
    TRILINOS_DISCRETIZATION
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

bob_option(CompadreHarness_DEBUG "Debug Mode" ON)
bob_option(Compadre_DEBUG "Debug Mode" "${CompadreHarness_DEBUG}")
option(Compadre_USE_Trilinos "" ON)
add_subdirectory(toolkit)

# RPATH related settings
# https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_RPATH})


bob_option(CompadreHarness_USE_Compose "Build compose for shape preservation and conservation post-processing" OFF)
if(CompadreHarness_USE_Compose)

  ##########
  #
  #
  #  COMPOSE LIBRARY (ONLY NEEDED IF HARNESS IS BEING BUILT)
  #
  #
  ##########
  
  set (Compose_Kokkos_DIR "${Trilinos_PREFIX}")
  set(Compose_PREFIX "${PROJECT_BINARY_DIR}/Compose-install")
  set(Compose_ARGS
    "-DCMAKE_INSTALL_PREFIX=${Compose_PREFIX}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DKokkos_DIR=${Compose_Kokkos_DIR}"
    "-DKOKKOS_IN_TRILINOS:BOOL=ON"
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
find_package(Doxygen QUIET)
if (DOXYGEN_FOUND)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in" "${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile" @ONLY)
  add_custom_target(doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc 
      VERBATIM 
  )
  set_target_properties(doc PROPERTIES EXCLUDE_FROM_ALL TRUE)
  # still must be called manually with "make doc"
endif()


set (augmented_list "${Trilinos_LIBRARIES}")
bob_end_package_no_recurse("${augmented_list}")
