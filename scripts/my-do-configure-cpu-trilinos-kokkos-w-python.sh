#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

MY_TRILINOS_PREFIX="/ascldap/users/pakuber/releases/Trilinos/build2/install"

cmake \
    -D CMAKE_CXX_COMPILER=mpic++ \
    -D CMAKE_INSTALL_PREFIX=./install \
    -D Compadre_USE_Trilinos:BOOL=ON \
    -D Compadre_USE_LAPACK:BOOL=ON \
    -D Compadre_USE_PYTHON:BOOL=ON \
    -D Trilinos_PREFIX:FILEPATH="$MY_TRILINOS_PREFIX" \
    \
    ..
