# ERROR MESSAGES IF A TEST IS NOT RUN
SET(SOLVER_SUPPORT_ISSUE "Missing Trilinos linear solver support.")

###############
#
#
# Template Functions
#
#
###############

function(add_exe_w_compadre EXE_NAME CPP_NAME)
  add_executable(${EXE_NAME} ${CPP_NAME})
  target_link_libraries(${EXE_NAME} PRIVATE physics compadreharness)
  bob_export_target(${EXE_NAME})
endfunction(add_exe_w_compadre)



###############
#
#
# CompadreHarness Examples / Tests
#
#
###############


if (CompadreHarness_TESTS)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../test_data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/..")
endif()

if (CompadreHarness_EXAMPLES)
  add_exe_w_compadre(CompadreGMLSStaggeredAnalyticTest Compadre_GMLSStaggeredAnalyticTest.cpp)
  add_exe_w_compadre(CompadreGMLSStaggeredIntegralTest Compadre_GMLSStaggeredIntegralTest.cpp)
  add_exe_w_compadre(CompadreGMLSCurlCurlTest Compadre_GMLSCurlCurlTest.cpp)
  add_exe_w_compadre(CompadreGMLSTest Compadre_GMLSTest.cpp)
  add_exe_w_compadre(CompadreGMLSPoissonNeumann Compadre_GMLSPoissonNeumannTest.cpp)
  add_exe_w_compadre(tpetraCoordsTest.exe Compadre_TrilinosTest.cpp)
  #add_exe_w_compadre(spherePoisson.exe Compadre_SpherePoissonTest.cpp)
  add_exe_w_compadre(kdtreeTest.exe Compadre_KdTreeTest.cpp)
  add_exe_w_compadre(readerTest.exe Compadre_SphReaderTest.cpp)
  add_exe_w_compadre(parallelReader.exe Compadre_ParallelReaderTest.cpp)
  add_exe_w_compadre(simpleCoordsTest.exe Compadre_SimpleCoordsUnitTest.cpp)
  add_exe_w_compadre(simpleMPITest.exe Compadre_SimpleMPIUnitTest.cpp)
  add_exe_w_compadre(remapTest.exe Compadre_RemapTest.cpp)
  add_exe_w_compadre(basisFunctionTest.exe Compadre_BasisFunction.cpp)
  add_exe_w_compadre(lagrangianTest.exe Compadre_LagrangianTest.cpp)
  add_exe_w_compadre(remoteRemapTest.exe Compadre_RemoteRemapTest.cpp)
  add_exe_w_compadre(laplaceBeltrami.exe Compadre_LaplaceBeltrami.cpp)
  add_exe_w_compadre(reactionDiffusion.exe Compadre_ReactionDiffusion.cpp)
  IF(CompadreHarness_CANGA_ENABLED)
    add_exe_w_compadre(cangaTestRemoteRemap.exe Canga_Test_RemoteRemap.cpp)
    add_exe_w_compadre(cangaTestIntercomparison.exe Canga_Test_Intercomparison.cpp)
    # no tests
    add_exe_w_compadre(cangaRemoteRemapSingleIter.exe Canga_RemoteRemap_SingleIter.cpp)
    add_exe_w_compadre(cangaRemoteRemapMultiIter.exe Canga_RemoteRemap_MultiIter.cpp)
    add_exe_w_compadre(cangaFileConverter.exe Canga_FileConverter.cpp)
  ENDIF()
  add_exe_w_compadre(lagrangianShallowWater.exe Compadre_LagrangianShallowWater.cpp)
  add_exe_w_compadre(RT0Remap.exe Compadre_RT0_Remap.cpp)
  add_exe_w_compadre(FVRemap.exe Compadre_FV_Remap.cpp)
  
  if (CompadreHarness_TESTS)

    # GMLS LAPLACIAN

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

    # GMLS STAGGERED ANALYTIC LAPLACIAN

    ADD_TEST(NAME GMLSStaggeredAnalyticTest4x1amg COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSStaggeredAnalyticTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_amg.xml" "--kokkos-threads=1")
    SET_TESTS_PROPERTIES(GMLSStaggeredAnalyticTest4x1amg PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;four;staggered")

    # GMLS STAGGERED INTEGRAL LAPLACIAN

    ADD_TEST(NAME GMLSStaggeredIntegralTest4x1amg COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSStaggeredIntegralTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_laplacian/parameters_amg.xml" "--kokkos-threads=1")
    SET_TESTS_PROPERTIES(GMLSStaggeredIntegralTest4x1amg PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;four;staggered")

    # GMLS CURL CURL TEST

    ADD_TEST(NAME CompadreGMLSCurlCurlPolyTest4x1amg COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSCurlCurlTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_curlcurl/parameters_amg.xml" "--kokkos-threads=1")
    SET_TESTS_PROPERTIES(CompadreGMLSCurlCurlPolyTest4x1amg PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;four")

    ADD_TEST(NAME CompadreGMLSCurlCurlSineTest4x1amg COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSCurlCurlTest "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_curlcurl/parameters_amg_sine.xml" "--kokkos-threads=1")
    SET_TESTS_PROPERTIES(CompadreGMLSCurlCurlSineTest4x1amg PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls;four")


    # GMLS LAPLACIAN WITH NEUMANN TEST

    ADD_TEST(NAME CompadreGMLSPoissonNeumann2x2amg COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/CompadreGMLSPoissonNeumann "--i=${CMAKE_CURRENT_BINARY_DIR}/../test_data/parameter_lists/gmls_neumann/parameters_amg.yml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(CompadreGMLSPoissonNeumann2x2amg PROPERTIES LABELS "ConvergenceTest;convergence;GMLS;gmls")

    # GRAPH LAPLACIAN

    ADD_TEST(NAME TrilinosTest COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/tpetraCoordsTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/trilinos_test/parameters_amg.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(TrilinosTest PROPERTIES LABELS "RegressionTest;regression;Trilinos;trilinos" TIMEOUT 180)
    
    ADD_TEST(NAME TrilinosTestBlockDiag5x5 COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/tpetraCoordsTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/trilinos_test/parameters_amg_blockdiag5x5.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(TrilinosTestBlockDiag5x5 PROPERTIES LABELS "RegressionTest;regression;Trilinos;trilinos;block;blocktl" TIMEOUT 180)


    # KDTREE TEST

    ADD_TEST(NAME KdTreeTest COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 1 ${CMAKE_CURRENT_BINARY_DIR}/kdtreeTest.exe "--kokkos-threads=1")
    SET_TESTS_PROPERTIES(KdTreeTest PROPERTIES LABELS "UnitTest;unit;nanoflann;KdTree" TIMEOUT 60)
    

    # READER TEST

    ADD_TEST(NAME SphReaderTest COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 3 ${CMAKE_CURRENT_BINARY_DIR}/readerTest.exe "${CMAKE_CURRENT_BINARY_DIR}/../test_data/grids/icos_tri_sphere/icosTri_1.nc" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(SphReaderTest PROPERTIES LABELS "UnitTest;unit;Reader" TIMEOUT 60)


    # PARALLEL READER TEST

    # named 2 because the files it will read in were written by two processors
    ADD_TEST(NAME ParallelReadNETCDFWriteNETCDFTest2 COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/parallelReader.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/reader_test/parameters_2_netcdf.xml" "--kokkos-threads=1")
    SET_TESTS_PROPERTIES(ParallelReadNETCDFWriteNETCDFTest2 PROPERTIES LABELS "UnitTest;unit;parallel;read;netcdf;Netcdf")
    
    # named 4 because the files it will read in were written by four processors
    ADD_TEST(NAME ParallelReadNETCDFWriteNETCDFTest4 COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/parallelReader.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/reader_test/parameters_4_netcdf.xml" "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(ParallelReadNETCDFWriteNETCDFTest4 PROPERTIES LABELS "UnitTest;unit;parallel;read;netcdf;Netcdf")


    # COORDS TEST

    ADD_TEST(NAME SimpleCoords COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/simpleCoordsTest.exe "--kokkos-threads=2")
    SET_TESTS_PROPERTIES(SimpleCoords PROPERTIES LABELS "UnitTest;unit;Coords" TIMEOUT 20)
    

    # MPI TEST

    ADD_TEST(NAME SimpleMPI COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/simpleMPITest.exe "--kokkos-threads=4")
    SET_TESTS_PROPERTIES(SimpleMPI PROPERTIES LABELS "UnitTest;unit;MPI" TIMEOUT 10)

    
    # REMAP TEST
    
    ADD_TEST(NAME RemapTestEulerianNanoflann COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/remapTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/remap/parameters_remap_eulerian_nano.xml" "--kokkos-threads=3")
    SET_TESTS_PROPERTIES(RemapTestEulerianNanoflann PROPERTIES LABELS "Regression;regression;Remap;remap;eulerian;Eulerian" TIMEOUT 180)
    
    ADD_TEST(NAME RemapTestLagrangianNanoflann COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/remapTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/remap/parameters_remap_lagrangian_nano.xml" "--kokkos-threads=3")
    SET_TESTS_PROPERTIES(RemapTestLagrangianNanoflann PROPERTIES LABELS "Regression;regression;Remap;remap;lagrangian;Lagrangian" TIMEOUT 180)
   
    # BASIS FUNCTION TEST

    ADD_TEST(NAME BasisFunction COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 1 ${CMAKE_CURRENT_BINARY_DIR}/basisFunctionTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/remap/parameters_basis.xml" "--kokkos-threads=3")
    SET_TESTS_PROPERTIES(BasisFunction PROPERTIES LABELS "basis;Regression;regression;Remap;remap;eulerian;Eulerian" TIMEOUT 180)
    
    # LAGRANGIAN POINT INSERTION REMAP TEST
    
    ADD_TEST(NAME TestLagrangianNanoflann COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/lagrangianTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/remap/parameters_remap_lagrangian_nano.xml" "--kokkos-threads=3")
    SET_TESTS_PROPERTIES(TestLagrangianNanoflann PROPERTIES LABELS "Regression;regression;lagrangian;Lagrangian;tln" TIMEOUT 180)
    

    # REMOTE REMAP TEST

    ADD_TEST(NAME RemoteRemapTest COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/remoteRemapTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/remap/parameters_remote_remap_lower.xml" "--kokkos-threads=1" ":" "-np" 3 ${CMAKE_CURRENT_BINARY_DIR}/remoteRemapTest.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/remap/parameters_remote_remap_upper.xml" "--kokkos-threads=1")
    SET_TESTS_PROPERTIES(RemoteRemapTest PROPERTIES LABELS "Regression;regression;Remap;remap;Remote;remote" TIMEOUT 180)
    

    # MANIFOLD OPERATOR CONVERGENCE TESTS

    ADD_TEST(NAME ManifoldTest COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/laplaceBeltrami.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/laplace_beltrami/parameters_basic.xml" "--kokkos-threads=1") # tests whatever is specified in parameters_basic
    SET_TESTS_PROPERTIES(ManifoldTest PROPERTIES LABELS "lapbel")
    
    ADD_TEST(NAME ManifoldScalarTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "0") # 4 polynomial order, 2 successive meshes, 0-scalar interpolation
    SET_TESTS_PROPERTIES(ManifoldScalarTest PROPERTIES LABELS "manifold;interpolation;scalar")
    
    ADD_TEST(NAME ManifoldVectorTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "1") # 4 polynomial order, 2 successive meshes, 1-scalar interpolation
    SET_TESTS_PROPERTIES(ManifoldVectorTest PROPERTIES LABELS "manifold;interpolation;vector")
    
    ADD_TEST(NAME ManifoldGradientTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "2") # 4 polynomial order, 2 successive meshes, 2-gradient interpolation
    SET_TESTS_PROPERTIES(ManifoldGradientTest PROPERTIES LABELS "manifold;interpolation;gradient")
    
    ADD_TEST(NAME ManifoldDivergenceOfGradientTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "3") # 4 polynomial order, 2 successive meshes, 3-divergence of gradient interpolation
    SET_TESTS_PROPERTIES(ManifoldDivergenceOfGradientTest PROPERTIES LABELS "manifold;interpolation;divergence;gradient")
    
    #ADD_TEST(NAME ManifoldStaggeredDivergenceOfGradientTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "4") # 4 polynomial order, 2 successive meshes, 4-divergence of gradient interpolation
    #SET_TESTS_PROPERTIES(ManifoldStaggeredDivergenceOfGradientTest PROPERTIES LABELS "manifold;interpolation;staggered;divergence;gradient")
    
    ADD_TEST(NAME ManifoldLaplaceTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "5") # 4 polynomial order, 2 successive meshes, 5-laplacian
    SET_TESTS_PROPERTIES(ManifoldLaplaceTest PROPERTIES LABELS "manifold;interpolation;laplace")
    
    ADD_TEST(NAME LaplaceBeltramiTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "6") # 4 polynomial order, 2 successive meshes, 6-laplace solve
    SET_TESTS_PROPERTIES(LaplaceBeltramiTest PROPERTIES LABELS "manifold;beltrami;Beltrami")
    
    #ADD_TEST(NAME ManifoldStaggeredGradientTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "7") # 4 polynomial order, 2 successive meshes, 7-staggered gradient interpolation
    #SET_TESTS_PROPERTIES(ManifoldStaggeredGradientTest PROPERTIES LABELS "manifold;interpolation;gradient;staggered")
    
    ADD_TEST(NAME ManifoldDivergenceTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "8") # 4 polynomial order, 2 successive meshes, 8-divergence interpolation
    SET_TESTS_PROPERTIES(ManifoldDivergenceTest PROPERTIES LABELS "manifold;interpolation;divergence;div;only")
    
    ADD_TEST(NAME ManifoldStaggeredDivergenceTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "9") # 4 polynomial order, 2 successive meshes, 9-staggered divergence interpolation
    SET_TESTS_PROPERTIES(ManifoldStaggeredDivergenceTest PROPERTIES LABELS "manifold;interpolation;divergence;div;only;staggered")
    
    ADD_TEST(NAME ManifoldStaggeredLaplaceTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/manifold_operator_convergence.py" "4" "2" "10") # 4 polynomial order, 2 successive meshes, 10-staggered laplacian
    SET_TESTS_PROPERTIES(ManifoldStaggeredLaplaceTest PROPERTIES LABELS "manifold;interpolation;laplacian;staggered")

    # ADVECTION DIFFUSION TESTS
    
    #ADD_TEST(NAME L2ProjectionTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/advectiondiffusion_convergence.py" "2" "1" "1" "0") # 4 polynomial order, 2 successive meshes, 1 l2 projection only, 0 - polynomial solution
    #SET_TESTS_PROPERTIES(L2ProjectionTest PROPERTIES LABELS "advection;diffusion;dg;variational;weakform;l2")

    #ADD_TEST(NAME L2ProjectionTestConv COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/advectiondiffusion_convergence.py" "2" "2" "1" "1") # 4 polynomial order, 2 successive meshes, 1 l2 projection only, 0 - polynomial solution
    #SET_TESTS_PROPERTIES(L2ProjectionTestConv PROPERTIES LABELS "advection;diffusion;dg;variational;weakform;l2")

    #ADD_TEST(NAME ReactionDiffusionTestConv COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/reactiondiffusion_convergence.py" "2" "3" "0" "1") # 4 polynomial order, 2 successive meshes, 1 l2 projection only, 0 - polynomial solution
    #SET_TESTS_PROPERTIES(ReactionDiffusionTestConv PROPERTIES LABELS "reaction;diffusion;dg;variational;weakform;")
    
    # CANGA TESTS
    
    IF (CompadreHarness_CANGA_ENABLED)
        ADD_TEST(NAME CangaTestRemoteRemap COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/cangaTestRemoteRemap.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/canga/parameters_lower.xml" "--kokkos-threads=1" ":" "-np" 5 ${CMAKE_CURRENT_BINARY_DIR}/cangaTestRemoteRemap.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/canga/parameters_upper.xml" "--kokkos-threads=1")
        SET_TESTS_PROPERTIES(CangaTestRemoteRemap PROPERTIES LABELS "Canga;canga;remote")
        
        ADD_TEST(NAME CangaConvergenceTest COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/sphere_convergence.py" "2" "2" "0") # 2nd polynomial order, 2 successive meshes, 0 for no obfet
        SET_TESTS_PROPERTIES(CangaConvergenceTest PROPERTIES LABELS "manifold;interpolation;canga")
        
        # This test uses OBFET, so it is limited to 2nd order asymptotically
        ADD_TEST(NAME CangaConvergenceTestWObfet COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../test_data/utilities/sphere_convergence.py" "2" "2" "1") # 2nd polynomial order, 2 successive meshes, 1 for obfet
        SET_TESTS_PROPERTIES(CangaConvergenceTestWObfet PROPERTIES LABELS "manifold;interpolation;canga;obfet")
        SET_TESTS_PROPERTIES(CangaConvergenceTestWObfet PROPERTIES SKIP_RETURN_CODE 77)

        ADD_TEST(NAME CangaTestIntercomparison COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/cangaTestIntercomparison.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/canga/parameters_lower_compare.xml" "--kokkos-threads=1" ":" "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/cangaTestIntercomparison.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/canga/parameters_upper_compare.xml" "--kokkos-threads=1")
        SET_TESTS_PROPERTIES(CangaTestIntercomparison PROPERTIES LABELS "Canga;canga;comparison;compare")
    ENDIF()
    
    IF (CompadreHarness_REBAR_ENABLED)
        
        # L2 projection
        ADD_TEST(NAME REBAR_L2_Exact_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=l2" "--solution=polynomial_2" "--num-meshes=1" "--penalty=1.0" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11")
        SET_TESTS_PROPERTIES(REBAR_L2_Exact_Test PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_L2_Convergence_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=l2" "--solution=sine" "--num-meshes=3" "--penalty=1.0" "--order=4" "--size=1.0")
        SET_TESTS_PROPERTIES(REBAR_L2_Convergence_Test PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Reaction Diffusion
        ADD_TEST(NAME REBAR_RD_Exact_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=rd" "--solution=polynomial_2" "--num-meshes=1" "--penalty=1.0" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-10")
        SET_TESTS_PROPERTIES(REBAR_RD_Exact_Test PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_RD_Convergence_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=rd" "--solution=sine" "--num-meshes=3" "--penalty=1.0" "--order=2" "--size=1.0")
        SET_TESTS_PROPERTIES(REBAR_RD_Convergence_Test PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Linear Elasticity
        ADD_TEST(NAME REBAR_LE_Exact_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=le" "--solution=polynomial_2" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=100.0")
        SET_TESTS_PROPERTIES(REBAR_LE_Exact_Test PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_LE_Convergence_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=le" "--solution=sine" "--num-meshes=4" "--order=2" "--size=1.0" "--penalty=100.0")
        SET_TESTS_PROPERTIES(REBAR_LE_Convergence_Test PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Vector Laplacian
        ADD_TEST(NAME REBAR_VL_Exact_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=vl" "--solution=polynomial_2" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=100.0")
        SET_TESTS_PROPERTIES(REBAR_VL_Exact_Test PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_VL_Convergence_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=vl" "--solution=sine" "--num-meshes=3" "--order=4" "--size=1.0" "--penalty=100.0")
        SET_TESTS_PROPERTIES(REBAR_VL_Convergence_Test PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Stokes Flow
        ADD_TEST(NAME REBAR_ST_Exact_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=st" "--solution=divfree_polynomial_2" "--pressure-solution=polynomial_1" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=10.0")
        SET_TESTS_PROPERTIES(REBAR_ST_Exact_Test PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_ST_Convergence_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=st" "--solution=divfree_sinecos" "--pressure-solution=polynomial_3" "--num-meshes=3" "--order=4" "--size=1.5" "--penalty=10.0")
        SET_TESTS_PROPERTIES(REBAR_ST_Convergence_Test PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Mixed Linear Elasticity
        ADD_TEST(NAME REBAR_MixLE_DivFree_Exact_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=mix_le" "--solution=divfree_polynomial_2" "--pressure-solution=polynomial_1" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=100.0")
        SET_TESTS_PROPERTIES(REBAR_MixLE_DivFree_Exact_Test PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_MixLE_Exact_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=mix_le" "--solution=polynomial_2" "--pressure-solution=polynomial_1" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=100.0")
        SET_TESTS_PROPERTIES(REBAR_MixLE_Exact_Test PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_MixLE_Convergence_Test COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=mix_le" "--solution=sine" "--num-meshes=3" "--order=3" "--size=1.5" "--penalty=100.0")
        SET_TESTS_PROPERTIES(REBAR_MixLE_Convergence_Test PROPERTIES LABELS "rebar;REBAR;Rebar")








        # L2 projection
        ADD_TEST(NAME REBAR_L2_Exact_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=l2" "--solution=polynomial_2" "--num-meshes=1" "--penalty=1.0" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_L2_Exact_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_L2_Convergence_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=l2" "--solution=sine" "--num-meshes=3" "--penalty=1.0" "--order=4" "--size=1.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_L2_Convergence_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Reaction Diffusion
        ADD_TEST(NAME REBAR_RD_Exact_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=rd" "--solution=polynomial_2" "--num-meshes=1" "--penalty=1.0" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-10" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_RD_Exact_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_RD_Convergence_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=rd" "--solution=sine" "--num-meshes=3" "--penalty=1.0" "--order=2" "--size=1.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_RD_Convergence_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Linear Elasticity
        ADD_TEST(NAME REBAR_LE_Exact_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=le" "--solution=polynomial_2" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=100.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_LE_Exact_Test PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_LE_Convergence_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=le" "--solution=sine" "--num-meshes=4" "--order=2" "--size=1.0" "--penalty=100.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_LE_Convergence_Test PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Vector Laplacian
        ADD_TEST(NAME REBAR_VL_Exact_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=vl" "--solution=polynomial_2" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=100.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_VL_Exact_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_VL_Convergence_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=vl" "--solution=sine" "--num-meshes=3" "--order=4" "--size=1.0" "--penalty=100.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_VL_Convergence_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Stokes Flow
        ADD_TEST(NAME REBAR_ST_Exact_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=st" "--solution=divfree_polynomial_2" "--pressure-solution=polynomial_1" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=10.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_ST_Exact_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_ST_Convergence_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=st" "--solution=divfree_sinecos" "--pressure-solution=polynomial_3" "--num-meshes=3" "--order=4" "--size=1.5" "--penalty=10.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_ST_Convergence_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")

        # Mixed Linear Elasticity
        ADD_TEST(NAME REBAR_MixLE_DivFree_Exact_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=mix_le" "--solution=divfree_polynomial_2" "--pressure-solution=polynomial_1" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=100.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_MixLE_DivFree_Exact_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_MixLE_Exact_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=mix_le" "--solution=polynomial_2" "--pressure-solution=polynomial_1" "--num-meshes=1" "--order=2" "--size=1.0" "--convergence-type=exact" "--rate-tol=1e-11" "--penalty=100.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_MixLE_Exact_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")
        ADD_TEST(NAME REBAR_MixLE_Convergence_Test_3D COMMAND "python" "${CMAKE_CURRENT_BINARY_DIR}/../../test_data/utilities/variational_convergence.py" "--operator=mix_le" "--solution=sine" "--num-meshes=3" "--order=3" "--size=1.5" "--penalty=100.0" "--dim=3" "--kokkos-threads=16")
        SET_TESTS_PROPERTIES(REBAR_MixLE_Convergence_Test_3D PROPERTIES LABELS "rebar;REBAR;Rebar")

    ENDIF()
    
    ## SHALLOW WATER TESTS
    #
    #ADD_TEST(NAME LagrangianShallowWater2 COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 2 ${CMAKE_CURRENT_BINARY_DIR}/lagrangianShallowWater.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/shallow_water/parameters_case2.xml" "--kokkos-threads=2")
    #SET_TESTS_PROPERTIES(LagrangianShallowWater2 PROPERTIES LABELS "shallow;case2")
    #
    #ADD_TEST(NAME LagrangianShallowWater5 COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/lagrangianShallowWater.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/shallow_water/parameters_case5.xml" "--kokkos-threads=2")
    #SET_TESTS_PROPERTIES(LagrangianShallowWater5 PROPERTIES LABELS "shallow;case5")


    ## RT0 REMAP TO VECTOR TEST
    #ADD_TEST(NAME RT0Remap COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/RT0Remap.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/remap/parameters_RT0.yml" "--kokkos-threads=2")
    #SET_TESTS_PROPERTIES(RT0Remap PROPERTIES LABELS "rt0")

    #ADD_TEST(NAME FVRemap COMMAND "mpirun" ${OPENMPI_FLAG} "-np" 4 ${CMAKE_CURRENT_BINARY_DIR}/FVRemap.exe "--i=${CMAKE_CURRENT_SOURCE_DIR}/../test_data/parameter_lists/remap/parameters_FV.yml" "--kokkos-threads=2")
    #SET_TESTS_PROPERTIES(FVRemap PROPERTIES LABELS "fv")

    # SET FLAGS FOR TESTS NOT RUN
    
    IF (NOT (TRILINOS_LINEAR_SOLVES))
      SET_TESTS_PROPERTIES(TrilinosTest PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(TrilinosTestBlockDiag5x5 PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(GMLSTest2x2PatchTest PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(GMLSTest2x2LagrangianPatchTest PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(GMLSTest2x2amg PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(GMLSTest2x2direct PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(GMLSTest2x2ifpackRelaxation PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(GMLSTest2x2ifpackRelaxationBlock1x1 PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(GMLSTest2x2ifpackILUT PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(GMLSTest4x1amg PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})

      SET_TESTS_PROPERTIES(GMLSStaggeredAnalyticTest4x1amg PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(GMLSStaggeredIntegralTest4x1amg PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})

      SET_TESTS_PROPERTIES(MeshWritePVTU PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(MeshWritePVTP PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
      SET_TESTS_PROPERTIES(LaplaceBeltramiTest PROPERTIES REQUIRED_FILES ${SOLVER_SUPPORT_ISSUE})
    ENDIF()

  endif()

ENDIF ()


bob_end_subdir()


