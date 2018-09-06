#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

# following lines for build directory cleanup
find . ! \( -name 'my-do-configure-cpu.sh' -o -name 'my-do-configure-gpu.sh' -o -name 'run_cpu_program.sh' -o -name 'run_gpu_program.sh' \) -type f -exec rm -f {} +
find . -mindepth 1 ! -name 'parameters' -type d -exec rm -rf {} +
mkdir test_output

cmake \
    -D CMAKE_CXX_COMPILER=mpic++ \
    -D Compadre_PREFIX=../build/install \
    -D EXAMPLE_TYPE="cpu" \
    \
    ..

# compadre prefix path is relative to .., not the current directory

