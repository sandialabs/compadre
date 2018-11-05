#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

MY_KOKKOSCORE_PREFIX="/ascldap/users/pakuber/releases/kokkos/launch/install"

cmake \
    -D CMAKE_CXX_COMPILER=/ascldap/users/pakuber/releases/kokkos/launch/install/bin/nvcc_wrapper \
    -D CMAKE_INSTALL_PREFIX=../../python-compadre-install \
    -D KokkosCore_PREFIX="$MY_KOKKOSCORE_PREFIX" \
    -D Compadre_USE_Trilinos:BOOL=OFF \
    -D Compadre_USE_PYTHON:BOOL=ON \
    \
    ..
