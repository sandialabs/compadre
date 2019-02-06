## COMPOSE: Compact Multi-moment Performance-Portable Semi-Lagrangian methods

COMPOSE provides libraries for semi-Lagrangian transport and, together or
separately, property preservation:

* CEDR: Communication-Efficient Constrained Density Reconstructors.
* SIQK: Spherical Polygon Intersection and Quadrature.

# Building and installing

First, install [Kokkos](https://github.com/kokkos/kokkos).
For example, in a typical environment using OpenMP, run:
```
    git clone https://github.com/kokkos/kokkos.git
    ./kokkos/generate_makefile.bash --with-serial --with-openmp --prefix=/path/to/my/kokkos/install --compiler=g++
    make -j8 install
```

Second, configure, build, and test COMPOSE:
```
    cmake \
        -D Kokkos_DIR=/path/to/my/kokkos/install \
        -D CMAKE_INSTALL_PREFIX=/path/to/my/compose/install \
        /path/to/compose/repo
    make -j8
    ctest
```

Optionally, third, install COMPOSE:
```
    make install
```

# Licence

COMPOSE version 1.0: Copyright 2018 National Technology & Engineering Solutions
of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.

This software is released under the BSD licence; see [LICENSE](./LICENSE).
