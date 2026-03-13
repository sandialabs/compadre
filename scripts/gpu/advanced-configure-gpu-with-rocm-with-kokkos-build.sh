#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

# Cuda on GPU via Kokkos
# With Python interface
# Standalone Kokkos will be auto-built

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

# pick your favorite c++ compiler
MY_CXX_COMPILER=/opt/rocm-6.4.3/bin/hipcc

# this will install in your build directory in a folder called install by default
INSTALL_PREFIX="./install" 

# optional CMake variables to pass in are PYTHON_PREFIX and PYTHON_EXECUTABLE
# if they are not passed in, then `which python` is called to determine your
# python executable, and from that sitepackages and libraries are inferred.
cmake \
    -D CMAKE_CXX_COMPILER="$MY_CXX_COMPILER" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D Compadre_USE_PYTHON:BOOL=OFF \
    -D Compadre_DEBUG:BOOL=OFF \
    -D Compadre_USE_HIP:BOOL=ON \
    -D Kokkos_ENABLE_HIP:BOOL=ON \
    -D Kokkos_ARCH_AMD_GFX942_APU:BOOL=ON \
    -D CMAKE_PREFIX_PATH:STRING="/opt/rocm-6.4.3" \
    -D CMAKE_BUILD_TYPE:STRING="Release" \
    \
    ..
