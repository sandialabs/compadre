#!/bin/bash
# 
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

rm -rf CMakeCache.txt CMakeFiles
MPI_ROOT=/usr/lib64/openmpi/
MPI_DIR=$MPI_ROOT
#NANOFLANN_ROOT=$HOME/compadre/development/nanoflann
BOOST_INC=/usr/include/boost148/boost
cmake \
    -D COMPADRE_TRILINOS_DIR:FILEPATH=$HOME/Trilinos/trilinos_install \
    -D USE_VTK:BOOL=OFF \
    -D CMAKE_C_COMPILER:FILEPATH=$MPI_DIR/bin/mpicc \
    -D CMAKE_CXX_COMPILER:FILEPATH=$MPI_DIR/bin/mpicxx \
    -D CMAKE_CXX_FLAGS="-O3 -g3 -fopenmp" \
    \
    .

#    -D COMPADRE_VTK_DIR:FILEPATH=$HOME/VTK/VTK-7.1.0/build/install \
