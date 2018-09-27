#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

cmake \
    -D CMAKE_CXX_COMPILER=mpic++ \
    -D CMAKE_INSTALL_PREFIX=./install \
    -D Compadre_USE_Trilinos:BOOL=ON \
    -D Compadre_USE_Trilinos_CXX_Flags:BOOL=ON \
    -D Compadre_USE_Trilinos_Solvers:BOOL=ON \
    -D Compadre_USE_Boost:BOOL=ON \
    -D Compadre_USE_Netcdf:BOOL=ON \
    -D Compadre_USE_VTK:BOOL=ON \
    -D Compadre_USE_PYTHON:BOOL=OFF \
    -D Trilinos_PREFIX:FILEPATH="../collaboration_Sandia_Compadre_atz/development/install" \
    -D Netcdf_LIBRARY:FILEPATH="/usr/lib/x86_64-linux-gnu/libnetcdf.so" \
    -D Netcdf_INCLUDE_DIRS:FILEPATH="/usr/include" \
    -D Boost_PREFIX:FILEPATH="/usr/include" \
    -D VTK_PREFIX:FILEPATH="../collaboration_Sandia_Compadre_atz/development/TPLs/VTK-8.1.1" \
    -D Compadre_EXAMPLES:BOOL=ON \
    -D Compadre_TESTS:BOOL=ON \
    -D GMLS_EXAMPLES:BOOL=ON \
    -D GMLS_TESTS:BOOL=ON \
    -D Compadre_CANGA_ENABLED:BOOL=ON \
    \
    ..
