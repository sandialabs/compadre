#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 

# following 3 lines for build directory cleanup
cp my-do-configure-cpu.sh my-do-configure-cpu.sh.gold 
find . ! \( -name 'my-do-configure-cpu.sh.gold' -o -name 'run_program.sh' \) -type f -exec rm -f {} +
find . -mindepth 1 ! -name 'parameters' -type d -exec rm -rf {} +
cp my-do-configure-cpu.sh.gold my-do-configure-cpu.sh

cmake \
    -D CMAKE_CXX_COMPILER=mpic++ \
    -D Compadre_PREFIX=../build/install \
    \
    ..

# compadre prefix path is relative to .., not the current directory

