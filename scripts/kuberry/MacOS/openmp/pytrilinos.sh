#!/bin/bash

SRC_DIR=/Users/pakuber/Compadre/pytrilinos
BUILD_DIR=`pwd`
INSTALL_DIR=$BUILD_DIR/install

rm -rf CMakeFiles/ CMakeCache.txt

cmake \
-Wno-dev \
-D CMAKE_C_COMPILER:FILEPATH=$MPICC \
-D CMAKE_CXX_COMPILER:FILEPATH=$MPICXX \
-D CMAKE_Fortran_COMPILER:FILEPATH=$MPIFC \
-D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \
-D CMAKE_BUILD_TYPE:STRING=RELEASE \
-D BUILD_SHARED_LIBS:BOOL=ON \
-D CMAKE_MACOSX_RPATH:BOOL=ON \
-D CMAKE_INSTALL_NAME_DIR:PATH=$INSTALL_DIR \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
\
-D 'MPI_EXEC_PRE_NUMPROCS_FLAGS:STRING=--mca;oob_tcp_if_include;lo0' \
-D Trilinos_ENABLE_TriUtils:BOOL=ON \
-D TPL_ENABLE_MPI:BOOL=ON \
-D TPL_ENABLE_Pthread:BOOL=OFF \
\
\
-D Trilinos_DISABLE_ENABLED_FORWARD_DEP_PACKAGES=ON \
-D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
-D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
\
-D Trilinos_ENABLE_Teuchos:BOOL=ON \
-D Teuchos_ENABLE_LONG_LONG_INT:BOOL=ON \
\
-D Trilinos_ENABLE_Tpetra:BOOL=OFF \
\
-D Trilinos_ENABLE_EpetraExt:BOOL=ON \
-D Trilinos_ENABLE_Isorropia:BOOL=ON \
-D Trilinos_ENABLE_Pliris:BOOL=ON \
-D Trilinos_ENABLE_AztecOO:BOOL=ON \
-D Trilinos_ENABLE_Galeri:BOOL=ON \
-D Trilinos_ENABLE_Amesos:BOOL=ON \
-D Trilinos_ENABLE_Komplex:BOOL=ON \
-D Trilinos_ENABLE_Anasazi:BOOL=ON \
-D Trilinos_ENABLE_ML:BOOL=ON \
-D Trilinos_ENABLE_MLAPI:BOOL=ON \
-D Trilinos_ENABLE_NOX:BOOL=OFF \
-D NOX_ENABLE_LOCA:BOOL=OFF \
-D Trilinos_ENABLE_PyTrilinos:BOOL=ON \
-D PyTrilinos_ENABLE_TESTS:BOOL=ON \
-D PyTrilinos_ENABLE_EXAMPLES:BOOL=ON \
-D PyTrilinos_INSTALL_PREFIX:PATH=/Users/pakuber \
\
-D Trilinos_ENABLE_Epetra:BOOL=ON \
-D Trilinos_ENABLE_AztecOO:BOOL=ON \
\
-D Trilinos_ENABLE_Kokkos:BOOL=OFF \
-D Trilinos_ENABLE_KokkosCore:BOOL=OFF \
-D Trilinos_ENABLE_KokkosAlgorithms:BOOL=OFF \
\
-D Trilinos_ENABLE_Belos:BOOL=ON \
-D Belos_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_Ifpack:BOOL=ON \
-D Ifpack_ENABLE_TESTS:BOOL=ON \
\
-D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
\
$SRC_DIR
