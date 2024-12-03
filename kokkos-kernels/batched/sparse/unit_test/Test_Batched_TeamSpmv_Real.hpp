//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, batched_scalar_team_spmv_nt_float_float) {
  typedef ::Test::Spmv::ParamTag<Trans::NoTranspose> param_tag_type;
  test_batched_team_spmv<TestDevice, float, float, param_tag_type>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, batched_scalar_team_spmv_nt_double_double) {
  typedef ::Test::Spmv::ParamTag<Trans::NoTranspose> param_tag_type;
  test_batched_team_spmv<TestDevice, double, double, param_tag_type>();
}
#endif
