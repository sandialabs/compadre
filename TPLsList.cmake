TRIBITS_REPOSITORY_DEFINE_TPLS(
  MPI       "${${PROJECT_NAME}_TRIBITS_DIR}/core/std_tpls/" PT
  BLAS      "${${PROJECT_NAME}_TRIBITS_DIR}/common_tpls/"   PT
  LAPACK    "${${PROJECT_NAME}_TRIBITS_DIR}/common_tpls/"   PT
  CUDA      "${${PROJECT_NAME}_TRIBITS_DIR}/core/std_tpls/" ST
  nanoflann "${CMAKE_CURRENT_SOURCE_DIR}/src/tpl/"          EX
  )
