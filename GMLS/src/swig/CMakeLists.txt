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

target_link_libraries(gmls_module PUBLIC gmls)
bob_link_dependency(gmls_module PUBLIC ${PYTHON_LIBRARIES})
target_include_directories(gmls_module PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(gmls_module PUBLIC ${Numpy_INCLUDE_DIRS})

bob_export_target(gmls_module)

CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in" "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/GMLS_Module.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")

install(FILES ${GMLS_Module_HEADERS} DESTINATION include)


install(TARGETS gmls_module
  ARCHIVE DESTINATION "${GMLS_Module_DEST}"
  LIBRARY DESTINATION "${GMLS_Module_DEST}"
)

set(GMLS_Module_PYTHON_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/GMLS_Module.py"
  "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
  )

install(FILES ${GMLS_Module_PYTHON_FILES} DESTINATION "${GMLS_Module_DEST}")

bob_end_subdir()

#  EXECUTE_PROCESS(
#  COMMAND "ldconfig" "-p"
#  COMMAND "grep" "stdc++"
#  OUTPUT_VARIABLE STDC_LIB
#  OUTPUT_STRIP_TRAILING_WHITESPACE )
#  STRING(REGEX REPLACE "(^[^\n\r]*x86-64\\) => )([^\n\r]*)(.*$)" "\\2" STDC_LIB ${STDC_LIB})

#  EXECUTE_PROCESS(
#  COMMAND "ldconfig" "-p"
#  COMMAND "grep" "gomp"
#  OUTPUT_VARIABLE GOMP_LIB
#  OUTPUT_STRIP_TRAILING_WHITESPACE )
#  STRING(REGEX REPLACE "(^[^\n\r]*x86-64\\) => )([^\n\r]*)(.*$)" "\\2" GOMP_LIB ${GOMP_LIB})
