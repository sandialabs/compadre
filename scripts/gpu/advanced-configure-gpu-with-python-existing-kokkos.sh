#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

# Cuda on GPU via Kokkos
# With Python interface
# Standalone Kokkos already built

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

# make sure you installed kokkos somewhere before setting this
# see doc/Kokkos-Install.md for details on installing Kokkos
MY_KOKKOSCORE_PREFIX="$HOME/releases/kokkos/install/pascal60"

# this will install in your build directory in a folder called install by default
INSTALL_PREFIX="./install"

# optional CMake variables to pass in are PYTHON_PREFIX and PYTHON_EXECUTABLE
# if they are not passed in, then `which python` is called to determine your
# python executable, and from that sitepackages and libraries are inferred.
cmake \
    -D CMAKE_CXX_COMPILER="$MY_KOKKOSCORE_PREFIX/bin/nvcc_wrapper" \
    -D KokkosCore_PREFIX="$MY_KOKKOSCORE_PREFIX" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D Compadre_USE_PYTHON:BOOL=ON \
    -D Compadre_USE_LAPACK:BOOL=OFF \
    -D Compadre_USE_CUDA:BOOL=ON \
    -D CUDA_PREFIX:PATH="/home/projects/ppc64le-pwr8-nvidia/cuda/9.2.88" \
    \
    ..

