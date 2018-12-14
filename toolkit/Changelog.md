Changelog {#changelog}
=============
Changes in version 0.0.2 (changes from 0.0.1)
================================
- Added documentation for GMLS class.
- Added storage of a batch of matrices in GMLS, which when combined with tags
  for functors, allowed for large code blocks to be broken up into smaller and 
  more readable functions.
- Use CuBLAS and CuSolverDN when on the GPU for QR and SVD factorizations via 
  a batched dgels and a batched SVD. We now use dgels and dgelsd when on the 
  CPU using LAPACK. This allowed for a significant reduction in the amount of
  code we need to support in GMLS_LinearAlgebra.
- Separated member functions of the GMLS class into GMLS.cpp, 
  GMLS_ApplyTargetEvaluations.hpp, GMLS_Basis.hpp, GMLS_Misc.hpp, 
  GMLS_Operators.hpp, GMLS_Quadrature.hpp, and GMLS_Targets.hpp.
- Removed all BOOST dependencies from the toolkit. Now, either building using 
  Kokkos on the GPU (Cuda) or providing LAPACK is required.
