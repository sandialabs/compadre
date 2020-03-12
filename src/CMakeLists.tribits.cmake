#
# A) Package-specific configuration options
#

tribits_configure_file(${PACKAGE_NAME}_Config.h)

#
# B) Define the header and source files (and directories)
#

set(HEADERS "")
set(SOURCES "")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(HEADERS ${HEADERS}
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_Config.h
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

append_glob(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
list(REMOVE_ITEM HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/Compadre_Manifold_Functions.hpp
  )
append_glob(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/basis/*.hpp)
append_glob(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/tpl/*.hpp)
append_glob(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

#
# C) Define the targets for package's library(s)
#

tribits_add_library(
  compadre
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )
