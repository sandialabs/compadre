#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 


# OpenMP on CPU via Kokkos
# With Python interface
# Standalone Kokkos

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

# Kokkos is built inside of trilinos
# this needs set to your Trilinos installation location
MY_TRILINOS_PREFIX="/path/to/your/trilinos/install"

# pick your favorite c++ compiler
MY_CXX_COMPILER=`which g++`

# this will install in your build directory in a folder called install by default
INSTALL_PREFIX="./install" 

# optional CMake variables to pass in are PYTHON_PREFIX and PYTHON_EXECUTABLE
# if they are not passed in, then `which python` is called to determine your
# python executable, and from that sitepackages and libraries are inferred.
cmake \
    -D CMAKE_CXX_COMPILER="$MY_CXX_COMPILER" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D Compadre_USE_Trilinos:BOOL=ON \
    -D Compadre_USE_LAPACK:BOOL=ON \
    -D Compadre_USE_PYTHON:BOOL=ON \
    -D Trilinos_PREFIX:FILEPATH="$MY_TRILINOS_PREFIX" \
    \
    ..
