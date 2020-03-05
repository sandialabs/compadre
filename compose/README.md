## COMPOSE: Compact multi-moment performance-portable semi-Lagrangian methods

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

# References

If you use COMPOSE, please cite
```
@misc{compose-software,
  title={{COMPOSE}: {C}ompact multi-moment performance-portable semi-{L}agrangian methods},
  author={A. M. Bradley and O. Guba and P. A. Bosler and M. A. Taylor},
  doi={10.5281/zenodo.2552888},
  howpublished={[Computer Software] \url{https://github.com/E3SM-Project/COMPOSE}},
  year={2019}
}
```
If you use CEDR in particular, please also cite
```
@article{compose-cedr,
  title={Communication-Efficient Property Preservation in Tracer Transport},
  author={A. M. Bradley and P. A. Bosler and O. Guba and M. A. Taylor and G. A. Barnett},
  journal={SIAM Journal on Scientific Computing},
  volume={41},
  number={3},
  pages={C161--C193},
  year={2019},
  publisher={SIAM},
  doi={10.1137/18M1165414}
}
```

# Licence

COMPOSE version 1.0: Copyright 2018 National Technology & Engineering Solutions
of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.

This software is released under the BSD licence; see [LICENSE](./LICENSE).
