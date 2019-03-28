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
# from https://raw.githubusercontent.com/ibaned/omega_h/master/cmake/detect_trilinos_opts.cmake

#if(DETECT_TRILINOS_OPTS)
#  return()
#endif()
#set(DETECT_TRILINOS_OPTS true)

function(detect_trilinos_opts)
  if (KokkosCore_FOUND)
    set(FILE_FOUND FALSE)
    foreach(INC_DIR IN LISTS KokkosCore_INCLUDE_DIRS)
      if (EXISTS "${INC_DIR}/KokkosCore_config.h")
        set(FILE_FOUND TRUE)
        message(STATUS "Found ${INC_DIR}/KokkosCore_config.h")
        file(READ "${INC_DIR}/KokkosCore_config.h" CONTENTS)
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_HAVE_CUDA")
          message(STATUS "KokkosCore has CUDA")
          set(KokkosCore_HAS_CUDA TRUE PARENT_SCOPE)
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_ENABLE_CUDA")
          message(STATUS "KokkosCore has CUDA")
          set(KokkosCore_HAS_CUDA TRUE PARENT_SCOPE)
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_HAVE_OPENMP")
          message(STATUS "KokkosCore has OpenMP")
          set(KokkosCore_HAS_OpenMP TRUE PARENT_SCOPE) 
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_ENABLE_OPENMP")
          message(STATUS "KokkosCore has OpenMP")
          set(KokkosCore_HAS_OpenMP TRUE PARENT_SCOPE)
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_HAVE_THREADS")
          message(STATUS "KokkosCore has Pthread")
          set(KokkosCore_HAS_Pthread TRUE PARENT_SCOPE) 
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_ENABLE_THREADS")
          message(STATUS "KokkosCore has Pthread")
          set(KokkosCore_HAS_Pthread TRUE PARENT_SCOPE)
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_HAVE_CUDA_LAMBDA")
          message(STATUS "KokkosCore has CUDA lambdas")
          set(KokkosCore_HAS_CUDA_LAMBDA TRUE PARENT_SCOPE)
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_CUDA_USE_LAMBDA")
          message(STATUS "KokkosCore has CUDA lambdas")
          set(KokkosCore_HAS_CUDA_LAMBDA TRUE PARENT_SCOPE)
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_ENABLE_CUDA_LAMBDA")
          message(STATUS "KokkosCore has CUDA lambdas")
          set(KokkosCore_HAS_CUDA_LAMBDA TRUE PARENT_SCOPE)
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_ENABLE_PROFILING_INTERNAL")
          message(STATUS "KokkosCore has profiling")
          set(KokkosCore_HAS_PROFILING TRUE PARENT_SCOPE)
        endif()
        if ("${CONTENTS}" MATCHES "\n\#define KOKKOS_ENABLE_PROFILING")
          message(STATUS "KokkosCore has profiling")
          set(KokkosCore_HAS_PROFILING TRUE PARENT_SCOPE)
        endif()
      endif()
    endforeach()
    if (NOT FILE_FOUND)
      message(FATAL_ERROR "Couldn't find KokkosCore_config.h")
    endif()
  endif()
  if (TeuchosComm_FOUND)
    list(FIND TeuchosComm_TPL_LIST "MPI" MPI_IDX)
    if (${MPI_IDX} GREATER -1)
      message(STATUS "TeuchosComm has MPI")
      set(TeuchosComm_HAS_MPI TRUE PARENT_SCOPE)
    endif()
  endif()
endfunction(detect_trilinos_opts)

