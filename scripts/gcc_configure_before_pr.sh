#!/bin/bash
# 
#
# Script for invoking CMake using the CMakeLists.txt file in this directory. 


# OpenMP on CPU via Kokkos
# No Python interface
# Standalone Kokkos

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

# pick your favorite c++ compiler
MY_CXX_COMPILER=/opt/homebrew/bin/g++-15

# this will install in your build directory in a folder called install by default
INSTALL_PREFIX="./install" 

cmake \
    -D CMAKE_CXX_COMPILER="$MY_CXX_COMPILER" \
    -D CMAKE_CXX_FLAGS=" -Wall -Wno-long-long -Wwrite-strings   -Wno-inline -Wno-deprecated-declarations -Werror=zero-length-bounds -Werror=write-strings -Werror=volatile-register-var -Werror=vla-parameter -Werror=variadic-macros -Werror=use-after-free=2 -Werror=unused-variable -Werror=unused-value -Werror=unused-local-typedefs -Werror=unused-label -Werror=unused-function -Werror=unused-const-variable=1 -Werror=unused-but-set-variable -Werror=unused-but-set-parameter -Werror=unused -Werror=unknown-pragmas -Werror=uninitialized -Werror=type-limits -Werror=trigraphs -Werror=tautological-compare -Werror=switch -Werror=string-compare -Werror=strict-overflow=1 -Werror=strict-aliasing -Werror=sizeof-pointer-memaccess -Werror=sizeof-pointer-div -Werror=sizeof-array-div -Werror=sized-deallocation -Werror=sign-compare -Werror=shadow -Werror=sequence-point -Werror=self-move -Werror=return-type -Werror=reorder -Werror=redundant-move -Werror=range-loop-construct -Werror=parentheses -Werror=packed-not-aligned -Werror=overloaded-virtual=1 -Werror=openmp-simd -Werror=nonnull-compare -Werror=nonnull -Werror=narrowing -Werror=multistatement-macros -Werror=missing-field-initializers -Werror=missing-attributes -Werror=mismatched-dealloc -Werror=misleading-indentation -Werror=memset-transposed-args -Werror=memset-elt-size -Werror=main -Werror=logical-not-parentheses -Werror=int-to-pointer-cast -Werror=int-in-bool-context -Werror=init-self -Werror=infinite-recursion -Werror=ignored-qualifiers -Werror=frame-address -Werror=format-truncation=1 -Werror=format-overflow=1 -Werror=format-diag -Werror=format=1 -Werror=format -Werror=expansion-to-defined -Werror=enum-compare -Werror=empty-body -Werror=div-by-zero -Werror=delete-non-virtual-dtor -Werror=dangling-reference -Werror=dangling-else -Werror=comment -Werror=clobbered -Werror=char-subscripts -Werror=catch-value -Werror=cast-function-type -Werror=cast-align -Werror=calloc-transposed-args -Werror=c++14-compat -Werror=c++11-compat -Werror=builtin-declaration-mismatch -Werror=bool-operation -Werror=bool-compare -Werror=array-compare -Werror=alloc-size -Werror=address -Wrestrict -Wno-error=restrict -Wpessimizing-move -Wno-error=pessimizing-move -Wmismatched-new-delete -Wno-error=mismatched-new-delete -Wmaybe-uninitialized -Wno-error=maybe-uninitialized -Wimplicit-fallthrough=3 -Wno-error=implicit-fallthrough=3 -Wdangling-pointer=2 -Wno-error=dangling-pointer=2 -Wclass-memaccess -Wno-error=class-memaccess -Warray-bounds=2 -Wno-error=array-bounds=2 -Waggressive-loop-optimizations -Wno-error=aggressive-loop-optimizations " \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -D Compadre_USE_PYTHON:BOOL=OFF \
    \
    ..
