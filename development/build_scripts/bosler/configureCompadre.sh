#!/bin/bash
# 
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

rm -rf CMakeCache.txt CMakeFiles/

TRILINOS_ROOT=$HOME/trilinos-openmp/install
NANOFLANN_ROOT=$HOME/compadre/development/nanoflann
VTK_ROOT=$HOME/VTK-8.1.0/build

cmake \
    -D CMAKE_BUILD_TYPE:STRING="RelWithDebInfo" \
    -D COMPADRE_TRILINOS_DIR:FILEPATH=$TRILINOS_ROOT \
    -D COMPADRE_VTK_DIR:FILEPATH=$VTK_ROOT \
    \
    ..
