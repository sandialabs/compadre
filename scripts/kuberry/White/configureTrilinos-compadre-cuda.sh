#!/bin/bash
# need to edit the file equivalent to /home/pakuber/releases/Trilinos/packages/kokkos/config/nvcc_wrapper
# so that it uses openmpi or mpich

SRC_DIR=/ascldap/users/pakuber/releases/Trilinos
BUILD_DIR=`pwd`
echo $BUILD_DIR
INSTALL_DIR=$SRC_DIR/install

rm -rf CMakeFiles/ CMakeCache.txt

cmake \
-Wno-dev \
-D CMAKE_C_COMPILER:FILEPATH=mpicc \
-D CMAKE_CXX_COMPILER:FILEPATH=/home/pakuber/releases/Trilinos/packages/kokkos/config/nvcc_wrapper \
\ #-D CMAKE_CXX_FLAGS:STRING="--std=c++14 -Wc++14-extensions -DKOKKOS_CUDA_USE_LAMBDA" \
-D CMAKE_CXX_FLAGS:STRING="  -O3 -w -DNDEBUG -std=c++11 --expt-extended-lambda --expt-relaxed-constexpr " \
-D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \
-D CMAKE_BUILD_TYPE:STRING=RELEASE \
-D BUILD_SHARED_LIBS:BOOL=OFF \
\ #-D CMAKE_MACOSX_RPATH:BOOL=ON \
-D CMAKE_INSTALL_NAME_DIR:PATH=$INSTALL_DIR \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-D CMAKE_MACOSX_RPATH:BOOL=OFF \
-D TPL_ENABLE_MPI:BOOL=ON \
-D Trilinos_ENABLE_OpenMP:BOOL=ON \
-D TPL_ENABLE_Pthread:BOOL=OFF \
-DCMAKE_CXX_FLAGS="-Wno-deprecated-gpu-targets -g -lineinfo -Xcudafe \
--diag_suppress=conversion_function_not_usable -Xcudafe \
--diag_suppress=cc_clobber_ignored -Xcudafe \
--diag_suppress=code_is_unreachable" \
-D TPL_ENABLE_CUDA:BOOL=ON \
-D Kokkos_ENABLE_Cuda_UVM:BOOL=ON \
\ #-DTPL_ENABLE_BLAS=OFF \
\ #-DTPL_ENABLE_LAPACK=OFF \
-DBLAS_LIBRARY_DIRS=/home/projects/pwr8-rhel73-lsf/openblas/0.2.19/gcc/5.3.0/lib \
-DLAPACK_LIBRARY_DIRS=/home/projects/pwr8-rhel73-lsf/openblas/0.2.19/gcc/5.3.0/lib \
\
\
-D Trilinos_DISABLE_ENABLED_FORWARD_DEP_PACKAGES=ON \
-D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
-D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
\
-D Trilinos_ENABLE_Teuchos:BOOL=ON \
-D Teuchos_ENABLE_LONG_LONG_INT:BOOL=ON \
\
-D Trilinos_ENABLE_Tpetra:BOOL=ON \
\
-D Trilinos_ENABLE_Kokkos:BOOL=ON \
-D Kokkos_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_KokkosCore:BOOL=ON \
-D Trilinos_ENABLE_KokkosAlgorithms:BOOL=ON \
-D Kokkos_ENABLE_Serial:BOOL=ON \
-D Kokkos_ENABLE_OpenMP:BOOL=ON \
-D Kokkos_ENABLE_Pthread:BOOL=OFF \
\
-D Trilinos_ENABLE_Zoltan2:BOOL=ON \
-D Zoltan2_ENABLE_Experimental:BOOL=ON\
-D Zoltan2_ENABLE_TESTS:BOOL=ON \
\
-D Trilinos_ENABLE_Belos:BOOL=ON \
-D Belos_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_Ifpack2:BOOL=ON \
-D Ifpack2_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_Amesos2:BOOL=ON \
-D Amesos2_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_EpetraExt:BOOL=ON \
-D Trilinos_ENABLE_MueLu:BOOL=ON \
-D MueLu_ENABLE_TESTS:BOOL=ON \
-D MueLu_ENABLE_EXAMPLES:STRING=ON \
\
-D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
-D Tpetra_INST_INT_LONG_LONG:BOOL=ON \
-D Tpetra_INST_INT_INT:BOOL=ON \
-D Tpetra_INST_DOUBLE:BOOL=ON \
-D Tpetra_INST_FLOAT:BOOL=OFF \
-D Tpetra_INST_COMPLEX_FLOAT:BOOL=OFF \
-D Tpetra_INST_COMPLEX_DOUBLE:BOOL=OFF \
-D Tpetra_INST_INT_LONG:BOOL=OFF \
-D Tpetra_INST_INT_UNSIGNED:BOOL=OFF \
-D Tpetra_INST_SERIAL:BOOL=ON \
\
$SRC_DIR

#\
#-D TPL_ENABLE_Netcdf:BOOL=ON \
#-D Netcdf_INCLUDE_DIRS:FILEPATH=$NETCDF_DIR/include \
#-D Netcdf_LIBRARY_DIRS:FILEPATH=$NETCDF_DIR/lib \
#\ #-DCMAKE_CXX_FLAGS="-ccbin clang++ -I../../common/inc  -m64  -Xcompiler -arch -Xcompiler x86_64  -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60 
#-DBLAS_LIBRARY_DIRS=
