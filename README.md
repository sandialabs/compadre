# COMpatible PArticle Discretization and REmap Toolkit

## About

The Compadre Toolkit provides a performance portable solution for the parallel evaluation of computationally dense kernels. The toolkit specifically targets the Generalized Moving Least Squares (GMLS) approach, which requires the inversion of small dense matrices. The result is a set of weights that provide the information needed for remap or entries that constitute the rows of some globally sparse matrix.

This toolkit focuses on the 'on-node' aspects of meshless PDE solution and remap, namely the parallel construction of small dense matrices and their inversion. What it does **not** provide is the tools for managing fields, inverting globally sparse matrices, or neighbor search that requires orchestration over many MPI processes. This toolkit is designed to be easily dropped-in to an existing MPI (or serial) based framework for PDE solution or remap, with minimal dependencies ([Kokkos](https://github.com/kokkos/kokkos) and either [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit) or [LAPACK](http://www.netlib.org/lapack/)).

### Generalized Moving Least Squares (GMLS)

A GMLS problem requires the specification of a target functional ![equation](https://latex.codecogs.com/gif.latex?\tau) (Compadre::TargetOperation), a reconstruction space ![equation](https://latex.codecogs.com/gif.latex?V) (Compadre::ReconstructionSpace), and a sampling functional ![equation](https://latex.codecogs.com/gif.latex?\lambda) (Compadre::SamplingFunctional).

The Compadre Toolkit is designed to efficiently assemble, factorize, and solve large batches of minimization problems having the form:

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Clarge%20%5C%5C%20%5Cbegin%7Balign*%7D%20p%5E%7B*%7D%26%20%3D%26%20%5Cunderset%7Bp%20%5Cin%20V%7D%7B%5Ctext%7Barg%20min%7D%7D%5C%3B%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bj%3D1%7D%5EN%20%28%5Clambda_j%28u%29-%5Clambda_j%28p%29%29%5E%7B2%7D%5Comega%28%5Ctau%3B%5Clambda_j%29%5C%5C%5C%5C%20%26%26%5Ctau%28u%29%20%5Capprox%20%5Ctau%28p%5E%7B*%7D%29%20%5Cend%7Balign*%7D)
<!---
https://www.codecogs.com/latex/eqneditor.php
\[\large \begin{align*}
p^{*}& =& \underset{p \in V}{\text{arg min}}\;\frac{1}{2}\sum_{j=1}^N (\lambda_j(u)-\lambda_j(p))^{2}\omega(\tau;\lambda_j)\\\\
&&\tau(u) \approx \tau(p^{*})
\end{align*} \]
--->

## Recent Changes
[Recent Changes](doc/Changelog.md)

## Installation
[Installation of Kokkos](doc/Kokkos-Install.md)

[Installation of Compadre](doc/Compadre-Install.md)

## Citing the Software

If you write a paper using results obtained with the help of the Compadre Toolkit, please cite the following reference:

```
@misc{paul_kuberry_2019_2560287,
  author       = {Paul Kuberry and
                  Peter Bosler and
                  Nathaniel Trask},
  title        = {Compadre Toolkit},
  month        = feb,
  year         = 2019,
  doi          = {10.5281/zenodo.2560287},
  url          = {https://doi.org/10.5281/zenodo.2560287}
}
```

If you would like to export the reference information to either CSL, DataCite, Dublin, Core, JSON, JSON-LD, MARCXML, or Mendeley, please find the export section at the bottom-right corner once you follow the link below:

<a href="https://doi.org/10.5281/zenodo.2557929" target="_blank"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.2557929.svg"></a>

