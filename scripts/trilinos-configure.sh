#!/bin/bash

###############################################################################
#
# USED FOR TESTING PURPOSES.  DO NOT LEAVE IN REPOSITORY.
#
###############################################################################
  
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

cmake \
  -D Kokkos_ENABLE_COMPLEX_ALIGN:BOOL=ON \
  -D Kokkos_ENABLE_OPENMP:BOOL=ON \
  -D Kokkos_ENABLE_SERIAL:BOOL=OFF \
  -D Trilinos_ENABLE_INSTALL_CMAKE_CONFIG_FILES:BOOL=ON \
  -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
  -D Trilinos_ENABLE_TESTS:BOOL=OFF \
  -D BUILD_SHARED_LIBS:BOOL=ON \
  -D Trilinos_ENABLE_DEBUG:BOOL=OFF \
  -D Trilinos_ENABLE_DEBUG_SYMBOLS:BOOL=ON \
  -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
  -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
  -D Trilinos_ENABLE_Compadre:BOOL=ON \
  -D Compadre_ENABLE_TESTS:BOOL=ON \
  -D Compadre_ENABLE_EXAMPLES:BOOL=ON \
  -D MPI_EXEC:FILEPATH="$(which mpiexec)" \
  -D CMAKE_CXX_COMPILER:FILEPATH="$(which mpicxx)" \
  -D CMAKE_C_COMPILER:FILEPATH="$(which mpicc)" \
  -D CMAKE_Fortran_COMPILER:FILEPATH="$(which mpif90)" \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
  -D CMAKE_SKIP_RULE_DEPENDENCY:BOOL=ON \
  -D Trilinos_VERBOSE_CONFIGURE:BOOL=ON \
  -D TPL_ENABLE_MPI:BOOL=ON \
  -D Trilinos_ENABLE_OpenMP:BOOL=ON \
  -D TPL_LAPACK_LIBRARIES:FILEPATH=/path/to/liblapack.so.3 \
  -D TPL_BLAS_LIBRARIES:FILEPATH=/path/to/libblas.so.3 \
  -D CMAKE_INSTALL_PREFIX:FILEPATH=/path/to/Trilinos/install \
  /path/to/Trilinos
###############################################################################
#
# ADJUST THE PATHS IN THE PREVIOUS FOUR LINES TO POINT TO THE APPROPRIATE
# PLACES
#
###############################################################################
