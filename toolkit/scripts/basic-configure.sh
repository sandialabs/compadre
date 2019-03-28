#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 


# OpenMP on CPU via Kokkos
# No Python interface
# Standalone Kokkos

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

# make sure you installed kokkos somewhere before setting this
# see doc/Kokkos-Install.md for details on installing Kokkos
MY_KOKKOSCORE_PREFIX="../../kokkos/build/install-openmp"

# pick your favorite c++ compiler
MY_CXX_COMPILER=`which g++`

# this will install in your build directory in a folder called install by default
INSTALL_PREFIX="./install" 

cmake \
    -D CMAKE_CXX_COMPILER="$MY_CXX_COMPILER" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D KokkosCore_PREFIX="$MY_KOKKOSCORE_PREFIX" \
    -D Compadre_USE_PYTHON:BOOL=OFF \
    \
    ..
