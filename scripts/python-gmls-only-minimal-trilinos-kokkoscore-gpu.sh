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
    -D Compadre_USE_Trilinos:BOOL=ON \
    -D Compadre_USE_Trilinos_CXX_Flags:BOOL=ON \
    -D Compadre_USE_Trilinos_Solvers:BOOL=OFF \
    -D Compadre_USE_Boost:BOOL=ON \
    -D Compadre_USE_Netcdf:BOOL=OFF \
    -D Compadre_USE_VTK:BOOL=OFF \
    -D Compadre_USE_PYTHON:BOOL=ON \
    -D Compadre_USE_Compadre:BOOL=OFF \
    -D Trilinos_PREFIX:FILEPATH="/ascldap/users/pakuber/releases/Trilinos/build3/install" \
    -D Netcdf_PREFIX:FILEPATH="/ascldap/users/pakuber/releases/netcdf" \
    -D Boost_PREFIX:FILEPATH="/usr/include" \
    -D VTK_PREFIX:FILEPATH="/ascldap/users/pakuber/releases/VTK" \
    -D Compadre_EXAMPLES:BOOL=OFF \
    -D Compadre_TESTS:BOOL=OFF \
    -D GMLS_EXAMPLES:BOOL=ON \
    -D GMLS_TESTS:BOOL=ON \
    -D Compadre_CANGA_ENABLED:BOOL=OFF \
    \
    ..
#    -D KokkosCore_PREFIX=/ascldap/users/pakuber/releases/kokkos/launch/install \
