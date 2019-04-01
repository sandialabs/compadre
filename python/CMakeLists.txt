#EXECUTE_PROCESS(
#  COMMAND "./run_swig.sh" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#)	
# If GMLS_Python.hpp is changed, then swig needs re-run and the resulting GMLS_Module.py and GMLS_Module_wrap.cxx updated in the repo

set(GMLS_Module_HEADERS
  GMLS_Python.hpp
  )

set(GMLS_Module_SOURCES
  GMLS_Module_wrap.cxx
  )



add_library(gmls_module ${GMLS_Module_SOURCES})
bob_library_includes(gmls_module)

SET_TARGET_PROPERTIES( gmls_module
  PROPERTIES
  OUTPUT_NAME "_GMLS_Module.so"
  PREFIX "" 
  SUFFIX "" 
)




# find all target dependencies

target_link_libraries(gmls_module PUBLIC compadre)
if (APPLE)
  set_target_properties(gmls_module PROPERTIES LINK_FLAGS "${CMAKE_PYTHON_SHARED_LINKER_FLAGS}")
endif()
target_include_directories(gmls_module PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(gmls_module PUBLIC ${Numpy_INCLUDE_DIRS})

# generate stdc++ and gomp variables and write them into python files 
# to ensure that correct libraries are linked

if(APPLE)

  EXECUTE_PROCESS(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from ctypes import util, cdll;so_name = util.find_library('stdc++');print(so_name);"
    OUTPUT_VARIABLE STDC_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE )
  EXECUTE_PROCESS(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from ctypes import util, cdll;so_name = util.find_library('gomp');print(so_name);"
    OUTPUT_VARIABLE GOMP_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE )

elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")

  EXECUTE_PROCESS(
  COMMAND "gcc" "-print-libgcc-file-name"
  OUTPUT_VARIABLE GCC_LIB_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE )

  GET_FILENAME_COMPONENT(GCC_LIB_PATH "${GCC_LIB_PATH}" DIRECTORY)

  FIND_LIBRARY(STDC_LIB NAMES stdc++ PATHS "${GCC_LIB_PATH}" NO_DEFAULT_PATH)
  FIND_LIBRARY(GOMP_LIB NAMES gomp PATHS "${GCC_LIB_PATH}" NO_DEFAULT_PATH)

elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")

  if (DEFINED ENV{MKL_ROOT}) 
    SET(MKL_ROOT "$ENV{MKL_ROOT}")
  elseif(DEFINED ENV{MKLROOT})
    SET(MKL_ROOT "$ENV{MKLROOT}")
  else()
    EXECUTE_PROCESS(
    COMMAND "which" "icc" 
    OUTPUT_VARIABLE ICC_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE )
    MESSAGE(STATUS "ICC ROOT: ${ICC_ROOT}")
    STRING(REGEX REPLACE "(^[^\n\r]*/)(icc$)" "\\1" ICC_ROOT ${ICC_ROOT})
    MESSAGE(STATUS "ICC ROOT: ${ICC_ROOT}")
    SET(MKL_ROOT "${ICC_ROOT}/../../mkl/lib/intel64")
    MESSAGE(STATUS "MKL ROOT: ${MKL_ROOT}")
  endif()

  # only needed if python needs to know mkl library locations
  bob_input(MKL_PREFIX "${MKL_ROOT}" PATH "Path to MKL libraries")
  if (MKL_PREFIX)
    set(CMAKE_PREFIX_PATH ${MKL_PREFIX} ${CMAKE_PREFIX_PATH})
    # libstdc++ will not conflict with anything in python
    # but additional libraries (MKL) are needed
    FIND_LIBRARY(MKL_CORE_LIB NAMES mkl_core)
    FIND_LIBRARY(MKL_MC3_LIB NAMES mkl_mc3)
    IF (NOT(MKL_CORE_LIB) OR NOT(MKL_MC3_LIB))
      MESSAGE(FATAL_ERROR "Compadre_USE_PYTHON:BOOL=ON, but using an Intel compiler without a valid MKL_PREFIX given.")
    ENDIF()
  else()
      MESSAGE(FATAL_ERROR "Compadre_USE_PYTHON:BOOL=ON, but using an Intel compiler without a valid MKL_PREFIX given.")
  endif()
  
  target_link_libraries(gmls_module PUBLIC ${MKL_CORE_LIB} ${MKL_MC3_LIB})

  EXECUTE_PROCESS(
  COMMAND "ldconfig" "-p"
  COMMAND "grep" "gomp"
  OUTPUT_VARIABLE GOMP_LIB
  OUTPUT_STRIP_TRAILING_WHITESPACE )
  STRING(REGEX REPLACE "(^[^\n\r]*x86-64\\) => )([^\n\r]*)(.*$)" "\\2" GOMP_LIB ${GOMP_LIB})

endif()


bob_export_target(gmls_module)

set(LOAD_CDLL "True")
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/compadre/__init__.py" "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" COPYONLY)
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/GMLS_Module.py.in" "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Module.py")

if(Compadre_USE_MATLAB)
  CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/compadre_py_util.py" "${CMAKE_CURRENT_BINARY_DIR}/compadre_py_util.py" COPYONLY)
  CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/Matlab_1D_Using_Python_Interface.m" "${CMAKE_CURRENT_BINARY_DIR}/Matlab_1D_Using_Python_Interface.m" COPYONLY)
endif()

install(FILES ${GMLS_Module_HEADERS} DESTINATION include)


install(TARGETS gmls_module
  ARCHIVE DESTINATION "${GMLS_Module_DEST}"
  LIBRARY DESTINATION "${GMLS_Module_DEST}"
  RUNTIME DESTINATION "${GMLS_Module_DEST}"
)

set(GMLS_Module_PYTHON_FILES
  "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Module.py"
  "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
  )

set(GMLS_Module_MATLAB_FILES
  "${CMAKE_CURRENT_BINARY_DIR}/compadre_py_util.py"
  "${CMAKE_CURRENT_BINARY_DIR}/Matlab_1D_Using_Python_Interface.m"
  )

install(FILES ${GMLS_Module_PYTHON_FILES} DESTINATION "${GMLS_Module_DEST}")
if(Compadre_USE_MATLAB)
    install(FILES ${GMLS_Module_MATLAB_FILES} DESTINATION "${GMLS_Module_DEST}")
endif()

bob_end_subdir()


