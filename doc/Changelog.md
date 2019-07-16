# Changelog
## Changes in version 1.0.2 (changes from 1.0.1)
- Kokkos source code now packaged in the toolkit, which is automatically configured and built. Users are still able to provide their own pre-built Kokkos install by specifying KokkosCore_PREFIX in CMake.
- Added support for additional evaluation sites, which are sites that a polynomial reconstruction is to be evaluated, but is not the target site about which the polynomial reconstruction was made.
- Exposed polynomial coefficients from the GMLS class through the Python interface.
- Fixed Pthread detection from Kokkos configuration.
- Added Python functions that generate neighbors lists and set the result in the GMLS class.
- Defined a global ordinal (long long int) and cast to it before doing any pointer arithmetic for tiled matrix starting locations.
- Added specification of a reference outward normal direction that is used to keep calculated outward normal vectors consistent [only relevant to manifold problems].
- Made memory usage more efficient by calculating the maximum number of neighbors rather than using the neighbor list matrix dimension size.
- Changed SamplingFunctionals from an ENUM with associated constexpr int arrays for characteristics to a constexpr struct that can be more easily extended by users.

## Changes in version 1.0.1 (changes from 1.0.0)
- Fixed README.md and added installation instructions as their own .md file

- Updated Python scripts to be compliant with Python 3.x

## Changes in version 1.0.0 (changes from 0.0.1)
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
