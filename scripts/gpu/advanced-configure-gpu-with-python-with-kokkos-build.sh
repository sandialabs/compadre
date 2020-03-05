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
MY_CXX_COMPILER=`which g++`

# this will install in your build directory in a folder called install by default
INSTALL_PREFIX="./install" 

# GPU specific 
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
NVCC_WRAPPER=$SCRIPTPATH/../kokkos/bin/nvcc_wrapper
NVCC_WRAPPER_DEFAULT_COMPILER=$MY_CXX_COMPILER

# optional CMake variables to pass in are PYTHON_PREFIX and PYTHON_EXECUTABLE
# if they are not passed in, then `which python` is called to determine your
# python executable, and from that sitepackages and libraries are inferred.
cmake \
    -D CMAKE_CXX_COMPILER="$NVCC_WRAPPER" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D Compadre_USE_PYTHON:BOOL=ON \
    -D Compadre_DEBUG:BOOL=OFF \
    -D Kokkos_ENABLE_PTHREAD:BOOL=OFF \
    -D Compadre_USE_CUDA:BOOL=ON \
    -D Kokkos_ARCH_PASCAL60:BOOL=ON \
    -D Kokkos_CUDA_DIR:PATH="/home/projects/ppc64le-pwr8-nvidia/cuda/9.2.88" \
    \
    ..
