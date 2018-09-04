# ERROR MESSAGES IF A TEST IS NOT RUN
SET(SOLVER_SUPPORT_ISSUE "Missing Trilinos linear solver support.")
SET(CANGA_ENABLED_ISSUE "Canga not enabled. Use CANGA_ENABLED:BOOL=ON to enable.")



if (Compadre_TESTS)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../test_data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/..")
endif()

if (Compadre_EXAMPLES)

  function(add_exe_w_trilinos EXE_NAME CPP_NAME)
    add_executable(${EXE_NAME} ${CPP_NAME})
    target_link_libraries(${EXE_NAME} PRIVATE physics compadre ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
    bob_export_target(${EXE_NAME})
  endfunction(add_exe_w_trilinos)
  
  
  #
  #	Executables
  #
  
  add_exe_w_trilinos(CompadreGMLSTest CompadreGMLSTest.cpp)


  if (Compadre_TESTS)

    ADD_TEST(NAME GMLSTest2x2PatchTest COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_patch.xml" "--kokkos-threads=3")
    SET_TESTS_PROPERTIES(GMLSTest2x2PatchTest PROPERTIES LABELS "Regression;regression;GMLS;gmls;Exact;exact;Patch;patch" TIMEOUT 180)
    
    ADD_TEST(NAME GMLSTest2x2LagrangianPatchTest COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_lagrangian_patch.xml" "--kokkos-threads=3")
    SET_TESTS_PROPERTIES(GMLSTest2x2LagrangianPatchTest PROPERTIES LABELS "Regression;regression;GMLS;gmls;Exact;exact;Patch;patch;lagrangian;Lagrangian" TIMEOUT 180)
    
    ADD_TEST(NAME GMLSTest2x2amg COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_amg.xml" "--kokkos-threads=3")
    SET_TESTS_PROPERTIES(GMLSTest2x2amg PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;amg" TIMEOUT 180)
    
    ADD_TEST(NAME GMLSTest2x2direct COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_direct.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLSTest2x2direct PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;direct")
    
    ADD_TEST(NAME GMLSTest2x2ifpackRelaxation COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_ifpack_relaxation.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLSTest2x2ifpackRelaxation PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;ifpack;relaxation;unb")
    
    ADD_TEST(NAME GMLSTest2x2ifpackRelaxationBlock1x1 COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_ifpack_relaxation_block1x1.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLSTest2x2ifpackRelaxationBlock1x1 PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;ifpack;relaxation;block")
    
    ADD_TEST(NAME GMLSTest2x2ifpackILUT COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_ifpack_ilut.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(GMLSTest2x2ifpackILUT PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;ifpack;ilut")
    
    ADD_TEST(NAME GMLSTest4x1amg COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_amg.xml" "--kokkos-threads=1")
    SET_TESTS_PROPERTIES(GMLSTest4x1amg PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;four")
    
    ADD_TEST(NAME MeshWritePVTU COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_mesh_pvtu.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(MeshWritePVTU PROPERTIES LABELS "UnitTest;unit;mesh;pvtu")
    
    ADD_TEST(NAME MeshWritePVTP COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_mesh_pvtp.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(MeshWritePVTP PROPERTIES LABELS "UnitTest;unit;mesh;pvtp")

  endif()


  add_exe_w_trilinos(tpetraCoordsTest.exe CompadreTrilinosTest.cpp)


  if (Compadre_TESTS)

    ADD_TEST(NAME TrilinosTest COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/tpetraCoordsTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/trilinos_test/parameters_amg.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(TrilinosTest PROPERTIES LABELS "RegressionTest;regression;Trilinos;trilinos" TIMEOUT 180)
    
    ADD_TEST(NAME TrilinosTestBlockDiag5x5 COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/tpetraCoordsTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/trilinos_test/parameters_amg_blockdiag5x5.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(TrilinosTestBlockDiag5x5 PROPERTIES LABELS "RegressionTest;regression;Trilinos;trilinos;block;blocktl" TIMEOUT 180)

  endif()

#ADD_EXECUTABLE(kdtreeTest.exe Compadre_KdTreeTest.cpp)
#TARGET_LINK_LIBRARIES(kdtreeTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(readerTest.exe Compadre_SphReaderTest.cpp)
#TARGET_LINK_LIBRARIES(readerTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(parallelReader.exe Compadre_ParallelReaderTest.cpp)
#TARGET_LINK_LIBRARIES(parallelReader.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(spherePoisson.exe Compadre_SpherePoissonTest.cpp)
#TARGET_LINK_LIBRARIES(spherePoisson.exe COMPADRE_LIBRARIES GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(simpleCoordsTest.exe Compadre_SimpleCoordsUnitTest.cpp)
#TARGET_LINK_LIBRARIES(simpleCoordsTest.exe COMPADRE_LIBRARIES)
#
#ADD_EXECUTABLE(simpleMPITest.exe Compadre_SimpleMPIUnitTest.cpp)
#TARGET_LINK_LIBRARIES(simpleMPITest.exe COMPADRE_LIBRARIES)
#
#ADD_EXECUTABLE(remapTest.exe Compadre_RemapTest.cpp)
#TARGET_LINK_LIBRARIES(remapTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(lagrangianTest.exe Compadre_LagrangianTest.cpp)
#TARGET_LINK_LIBRARIES(lagrangianTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#ADD_EXECUTABLE(remoteRemapTest.exe Compadre_RemoteRemapTest.cpp)
#TARGET_LINK_LIBRARIES(remoteRemapTest.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#IF(Compadre_CANGA_ENABLED)
#  ADD_EXECUTABLE(cangaRemoteRemap.exe Canga_RemoteRemap.cpp)
#  TARGET_LINK_LIBRARIES(cangaRemoteRemap.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#
#  ADD_EXECUTABLE(cangaIntercomparison.exe Canga_Intercomparison.cpp)
#  TARGET_LINK_LIBRARIES(cangaIntercomparison.exe COMPADRE_LIBRARIES ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} GMLS_LIBRARIES)
#ENDIF()















ENDIF (Compadre_EXAMPLES)


bob_end_subdir()


