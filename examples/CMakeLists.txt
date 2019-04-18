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

  if(Compadre_USE_LAPACK)
    add_exe_w_compadre(LAPACK_Test UnitTest_ThreadedLapack.cpp) 
  endif()
  add_exe_w_compadre(GMLS_Host_Test GMLS_Host.cpp)
  add_exe_w_compadre(GMLS_Device_Test GMLS_Device.cpp)
  add_exe_w_compadre(GMLS_Vector_Test GMLS_Vector.cpp)
  add_exe_w_compadre(GMLS_SmallBatchReuse_Device_Test GMLS_SmallBatchReuse_Device.cpp)
  add_exe_w_compadre(GMLS_Manifold_Test GMLS_Manifold.cpp)
  add_exe_w_compadre(GMLS_Staggered_Manifold_Test GMLS_Staggered_Manifold.cpp)
  add_exe_w_compadre(GMLS_MultiSite_Test GMLS_Multiple_Evaluation_Sites.cpp)

  if (Compadre_TESTS)

    if(Compadre_USE_LAPACK)
      # Test if LAPACK+BLAS are compatible for use in the toolkit
      ADD_TEST(NAME LAPACK_THREADSAFE COMMAND ${CMAKE_CURRENT_BINARY_DIR}/LAPACK_Test "--kokkos-threads=8" --verbose --output-on-failure)
      SET_TESTS_PROPERTIES(LAPACK_THREADSAFE PROPERTIES LABELS "UnitTest;unit;lapack;Lapack;LAPACK;kokkos" TIMEOUT 5)
      SET_TESTS_PROPERTIES(LAPACK_THREADSAFE PROPERTIES SKIP_RETURN_CODE 77)
    endif()

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

    # Device views tests for GMLS (vector basis)
    ADD_TEST(NAME GMLS_Vector_Dim3_QR COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Device_Test "3" "20" "3" "1" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_Vector_Dim3_QR PROPERTIES LABELS "UnitTest;unit;kokkos;vector" TIMEOUT 10)
    
    ADD_TEST(NAME GMLS_Vector_Dim2_QR COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Device_Test "3" "20" "2" "1" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_Vector_Dim2_QR PROPERTIES LABELS "UnitTest;unit;kokkos;vector" TIMEOUT 10)
    
    ADD_TEST(NAME GMLS_Vector_Dim1_QR COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Device_Test "3" "20" "1" "1" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_Vector_Dim1_QR PROPERTIES LABELS "UnitTest;unit;kokkos;vector" TIMEOUT 10)

    # Device views tests for small batch GMLS, reusing GMLS class object
    ADD_TEST(NAME GMLS_SmallBatchReuse_Device_Dim2_QR COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_SmallBatchReuse_Device_Test "4" "200" "2" "1" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_SmallBatchReuse_Device_Dim2_QR PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)
    
    ADD_TEST(NAME GMLS_SmallBatchReuse_Device_Dim1_QR COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_SmallBatchReuse_Device_Test "4" "200" "1" "1" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_SmallBatchReuse_Device_Dim1_QR PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)

    # Multisite test for GMLS
    ADD_TEST(NAME GMLS_MultiSite_Dim3_QR COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_MultiSite_Test "4" "200" "3" "1" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLS_MultiSite_Dim3_QR PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)

    #    # Manifold tests for GMLS
    #    ADD_TEST(NAME GMLS_Manifold_Dim3 COMMAND ${CMAKE_CURRENT_BINARY_DIR}/GMLS_Manifold_Test "4" "200" "3" "1" "--kokkos-threads=2")
    #    SET_TESTS_PROPERTIES(GMLS_Manifold_Dim3 PROPERTIES LABELS "UnitTest;unit;kokkos" TIMEOUT 10)

    # Python driven test of a C++ executable (Python changes command line arguments given to executable)
    CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/GMLS_Manifold.py.in" "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Manifold.py" @ONLY)
    ADD_TEST(NAME GMLS_Manifold_Refinement_Study COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Manifold.py" "3" "4")
    SET_TESTS_PROPERTIES(GMLS_Manifold_Refinement_Study PROPERTIES LABELS "ConvergenceTest;convergence;manifold" TIMEOUT 20)

    # Python driven test of a C++ executable (Python changes command line arguments given to executable)
    CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/GMLS_Staggered_Manifold.py.in" "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Staggered_Manifold.py" @ONLY)
    ADD_TEST(NAME GMLS_Staggered_Manifold_Refinement_Study COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/GMLS_Staggered_Manifold.py" "3" "4")
    SET_TESTS_PROPERTIES(GMLS_Staggered_Manifold_Refinement_Study PROPERTIES LABELS "ConvergenceTest;convergence;manifold;staggered" TIMEOUT 20)

    # WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
    if (Compadre_USE_PYTHON)
      CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/Python_3D_Convergence.py.in" "${CMAKE_CURRENT_BINARY_DIR}/Python_3D_Convergence.py" @ONLY)
      ADD_TEST(NAME GMLS_Python_Convergence_Test_3d_Point_Reconstruction COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/Python_3D_Convergence.py")
      SET_TESTS_PROPERTIES(GMLS_Python_Convergence_Test_3d_Point_Reconstruction PROPERTIES LABELS "UnitTest;unit;python;kokkos" TIMEOUT 10)

      if (Compadre_USE_MATLAB)
        ADD_TEST(NAME GMLS_Matlab_Python_Interface_1d_Point_Reconstruction COMMAND "matlab" "-nodisplay" "-nojvm" "-nosplash" "-nodesktop" "-r \"try, run('${SWIG_PREFIX}/Matlab_1D_Using_Python_Interface.m'), catch, exit(1), end, exit(0);\"")
        SET_TESTS_PROPERTIES(GMLS_Matlab_Python_Interface_1d_Point_Reconstruction PROPERTIES LABELS "UnitTest;unit;python;kokkos;matlab" TIMEOUT 10)
      endif()
    endif()

  endif (Compadre_TESTS)
endif (Compadre_EXAMPLES)

bob_end_subdir()


