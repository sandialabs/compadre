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
MY_KOKKOS_ROOT="$HOME/releases/kokkos/install/ampere80"
MY_KOKKOSKERNELS_ROOT="$HOME/releases/kokkos/install/ampere80"

# this will install in your build directory in a folder called install by default
INSTALL_PREFIX="./install"

# optional CMake variables to pass in are PYTHON_PREFIX and PYTHON_EXECUTABLE
# if they are not passed in, then `which python` is called to determine your
# python executable, and from that sitepackages and libraries are inferred.
cmake \
    -D CMAKE_CXX_COMPILER="$MY_KOKKOSCORE_PREFIX/bin/nvcc_wrapper" \
    -D Kokkos_ROOT="$MY_KOKKOS_ROOT" \
    -D KokkosKernels_ROOT="$MY_KOKKOSKERNELS_ROOT" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D CMAKE_BUILD_TYPE:STRING="Release" \
    -D Compadre_USE_PYTHON:BOOL=OFF \
    -D Kokkos_ENABLE_CUDA:BOOL=ON \
    -D Kokkos_ARCH_AMPERE80:BOOL=ON \
    -D Kokkos_ENABLE_CUDA_CONSTEXPR:BOOL=ON \
    -D CUDA_ROOT:PATH="/home/projects/cuda/12.9.1" \
    \
    ..

