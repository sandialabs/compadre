# Copyright 2017 Sandia Corporation.
# Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
# the U.S. Government retains certain rights in this software.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of Sandia Corporation.
# 
# from https://github.com/sandialabs/omega_h/blob/main/cmake/detect_kokkos_opts.cmake

if(DETECT_KOKKOS_OPTS)
  return()
endif()
set(DETECT_KOKKOS_OPTS true)

function(detect_kokkos_opts)
  if (KokkosCore_FOUND)
    list(FIND Kokkos_DEVICES SERIAL serial_index)
    list(FIND Kokkos_DEVICES OPENMP openmp_index)
    list(FIND Kokkos_DEVICES THREADS threads_index)
    list(FIND Kokkos_DEVICES PTHREAD pthread_index)
    list(FIND Kokkos_DEVICES CUDA cuda_index)
    if (NOT cuda_index EQUAL -1)
      message(STATUS "Kokkos has CUDA")
      set(KokkosCore_HAS_CUDA TRUE PARENT_SCOPE)
      set(KokkosCore_HAS_CUDA_LAMBDA TRUE PARENT_SCOPE)
    endif()
    if (NOT openmp_index EQUAL -1)
      message(STATUS "Kokkos has OpenMP")
      set(KokkosCore_HAS_OPENMP TRUE PARENT_SCOPE)
    endif()
    if ((NOT threads_index EQUAL -1) OR (NOT pthread_index EQUAL -1))
      message(STATUS "Kokkos has Threads")
      set(KokkosCore_HAS_THREADS TRUE PARENT_SCOPE)
    endif()
    if (NOT serial_index EQUAL -1)
      message(STATUS "Kokkos has Serial")
      set(KokkosCore_HAS_SERIAL TRUE PARENT_SCOPE)
    endif()
  endif()
endfunction(detect_kokkos_opts)

