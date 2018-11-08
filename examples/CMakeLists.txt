# ERROR MESSAGES IF A TEST IS NOT RUN
SET(OPENBLAS_NUM_THREADS "1")


###############
#
#
# Template Functions
#
#
###############

function(add_exe_w_compadre EXE_NAME CPP_NAME)
  add_executable(${EXE_NAME} ${CPP_NAME})
  target_link_libraries(${EXE_NAME} PRIVATE compadre)
  bob_export_target(${EXE_NAME})
endfunction(add_exe_w_compadre)

###############
#
#
# GMLS Examples / Tests
#
#
###############
if (Compadre_EXAMPLES)

  add_exe_w_compadre(GMLS_Host_Test GMLS_Host.cpp)
  add_exe_w_compadre(GMLS_Device_Test GMLS_Device.cpp)
  add_exe_w_compadre(GMLS_Manifold_Test GMLS_Manifold.cpp)

  if (Compadre_TESTS)

    # Host views tests for GMLS
    ADD_TEST(NAME GMLS_Host_Dim3_SVD COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Host_Test "4" "200" "3" "0" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_Host_Dim3_SVD PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)
    
    ADD_TEST(NAME GMLS_Host_Dim2_SVD COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Host_Test "4" "200" "2" "0" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_Host_Dim2_SVD PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)
    
    ADD_TEST(NAME GMLS_Host_Dim1_SVD COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Host_Test "4" "200" "1" "0" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_Host_Dim1_SVD PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)

    # Device views tests for GMLS
    ADD_TEST(NAME GMLS_Device_Dim3_QR COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Device_Test "4" "200" "3" "1" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_Device_Dim3_QR PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)
    
    ADD_TEST(NAME GMLS_Device_Dim2_QR COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Device_Test "4" "200" "2" "1" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_Device_Dim2_QR PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)
    
    ADD_TEST(NAME GMLS_Device_Dim1_QR COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Device_Test "4" "200" "1" "1" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_Device_Dim1_QR PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)

    #    # Manifold tests for GMLS
    #    ADD_TEST(NAME GMLS_Manifold_Dim3 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Manifold_Test "4" "200" "3" "1" "--kokkos-threads=2")
    #    SET_TESTS_PROPERTIES(GMLS_Manifold_Dim3 PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)

    # Python driven test of a C++ executable (Python changes command line arguments given to executable)
    CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/GMLS_Manifold.py.in" "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Manifold.py" @ONLY)
    ADD_TEST(NAME GMLS_Manifold_Refinement_Study COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Manifold.py" "3" "4")
    SET_TESTS_PROPERTIES(GMLS_Manifold_Refinement_Study PROPERTIES LABELS "ConvergenceTest;convergence;manifold" TIMEOUT 10)

    # WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
    if (Compadre_USE_PYTHON)
      CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/gmls_3d_convergence.py.in" "${CMAKE_CURRENT_BINARY_DIR}/gmls_3d_convergence.py" @ONLY)
      ADD_TEST(NAME GMLS_Python_Convergence_Test_3d_Point_Reconstruction COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 1 ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/gmls_3d_convergence.py")
      SET_TESTS_PROPERTIES(GMLS_Python_Convergence_Test_3d_Point_Reconstruction PROPERTIES LABELS "UnitTest;unit;python;kokkos" TIMEOUT 10)
    endif()

  endif (Compadre_TESTS)
endif (Compadre_EXAMPLES)

bob_end_subdir()


