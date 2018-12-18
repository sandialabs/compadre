#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

cmake \
    -D CMAKE_CXX_COMPILER=/ascldap/users/pakuber/releases/kokkos/launch/install/bin/nvcc_wrapper \
    -D CMAKE_INSTALL_PREFIX=./install \
    -D CompadreHarness_USE_Trilinos:BOOL=ON \
    -D CompadreHarness_USE_Trilinos_CXX_Flags:BOOL=ON \
    -D CompadreHarness_USE_Trilinos_Solvers:BOOL=OFF \
    -D CompadreHarness_USE_Netcdf:BOOL=OFF \
    -D CompadreHarness_USE_VTK:BOOL=OFF \
    -D CompadreHarness_USE_PYTHON:BOOL=ON \
    -D CompadreHarness_USE_Harness:BOOL=OFF \
    -D CompadreHarness_EXAMPLES:BOOL=OFF \
    -D CompadreHarness_TESTS:BOOL=OFF \
    -D CompadreHarness_CANGA_ENABLED:BOOL=OFF \
    -D Toolkit_EXAMPLES:BOOL=ON \
    -D Toolkit_TESTS:BOOL=ON \
    -D Trilinos_PREFIX:FILEPATH="/ascldap/users/pakuber/releases/Trilinos/build3/install" \
    -D Netcdf_PREFIX:FILEPATH="/ascldap/users/pakuber/releases/netcdf" \
    -D VTK_PREFIX:FILEPATH="/ascldap/users/pakuber/releases/VTK" \
    \
    ..
