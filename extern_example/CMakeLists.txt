cmake_minimum_required(VERSION 3.0.0)

##########
#
#   Test
#
########## 
PROJECT(Test VERSION 0.0.1 LANGUAGES CXX)

#COMPADRE
set(Test_USE_Compadre "Use Compadre" ON)
if (Compadre_PREFIX)
  set(CMAKE_PREFIX_PATH ${Compadre_PREFIX} ${CMAKE_PREFIX_PATH})
endif()
IF (Test_USE_Compadre)
  FIND_PACKAGE(Compadre)
  IF (Compadre_FOUND)
  	MESSAGE("\nFound Compadre!  Here are the details: ")
  	MESSAGE("   Compadre_VERSION = ${Compadre_VERSION}")
  	MESSAGE("   Compadre_EXPORTED_TARGETS = ${Compadre_EXPORTED_TARGETS}")
  ELSE()
    MESSAGE(FATAL_ERROR "Compadre not found, but requested because Test_USE_Compadre=ON")
  ENDIF()
ENDIF()
set(CMAKE_CXX_FLAGS ${Compadre_CXX_FLAGS} ${CMAKE_CXX_FLAGS}) 


if (("${EXAMPLE_TYPE}" STREQUAL "cpu") OR ("${EXAMPLE_TYPE}" STREQUAL "gpu"))
  # Copy example .cpp problem from source directory (not needed for your program)
  # destination path is relative to directory calling CMake (i.e. ./build)
  file(COPY ../development/examples/GMLS_PointValuesTest_Kokkos.cpp
          DESTINATION .. )
  add_executable(GMLS_PointValuesTest_Kokkos GMLS_PointValuesTest_Kokkos.cpp)
  target_link_libraries(GMLS_PointValuesTest_Kokkos Compadre::gmls)
endif()

if ("${EXAMPLE_TYPE}" STREQUAL "cpu") 
  # Copy example .cpp problem from source directory (not needed for your program)
  # destination path is relative to directory calling CMake (i.e. ./build)
  file(COPY ../development/examples/CompadreTrilinosTest.cpp
          DESTINATION .. )
  add_executable(tpetraCoordsTest.exe CompadreTrilinosTest.cpp)
  target_link_libraries(tpetraCoordsTest.exe Compadre::compadre Compadre::physics)
endif()

