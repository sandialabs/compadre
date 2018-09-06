#!/bin/bash
# 
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

rm -rf CMakeCache.txt CMakeFiles
MPI_ROOT=/Users/pakuber/Compadre/openmpi-2.0.2/build/install
MPI_DIR=$MPI_ROOT

cmake \
    -D COMPADRE_TRILINOS_DIR:FILEPATH=/Users/pakuber/Compadre/trilinos/build/install \
    -D COMPADRE_VTK_DIR:FILEPATH=/usr/local/opt/vtk \
    -D CMAKE_C_COMPILER:FILEPATH=$MPI_DIR/bin/mpicc \
    -D CMAKE_CXX_COMPILER:FILEPATH=$MPI_DIR/bin/mpicxx \
    -D CMAKE_CXX_FLAGS="-O3 -g3 -fopenmp" \
    .
