# Installing the Compadre Harness
 
## Compadre Harness Installation Instructions

### Requirements

  **Compadre Harness includes the Compadre Toolkit as a subdirectory, which will be configured and built while installing the Harness.

  **All build instructions are only tested on linux platforms.

  NetCDF is the means by which files can be read into the Harness. It is expected that a user will have it installed, but it can be disabled in the CMake configure script using the variable


```
-D CompadreHarness_USE_Netcdf:BOOL=OFF
```

  For VTK (.vtk, .pvtu, and .pvtp) type files:
    If you would like your input/output in a VTK style, there are two utilities `./test_data/utilities/convert_nc_to_vtk.py`, and `./test_data/utilities/convert_vtk_to_nc.py` for pre and post processing. Directions on their use are contained inside of the .py files. They both require that a user have the python package for 'vtk' and 'netCDF4' installed.


#### Trilinos

  The Compadre Harness requires Trilinos with a certain set of packages in Trilinos turned on. Example scripts are provided which turn on the needed packages in the Trilinos installation section.


 1.)  Get Trilinos source at: https://github.com/trilinos/Trilinos 

 2.)  After cloning, go into Trilinos directory followed by:
```
    >> mkdir build
    >> cd build
```
 3.) then run this script (with your changes to `SRC_DIR`, `MPI_DIR`, and `INSTALL_DIR`):

```
#!/bin/bash

SRC_DIR=/your/Trilinos-source
BUILD_DIR=$SRC_DIR/build
INSTALL_DIR=/your/Trilinos-install
MPI_DIR=/your/mpi-base-dir

rm -rf CMakeFiles/ CMakeCache.txt

cmake \
-D CMAKE_C_COMPILER:FILEPATH=$MPI_DIR/bin/mpicc \
-D CMAKE_CXX_COMPILER:FILEPATH=$MPI_DIR/bin/mpicxx \
-D CMAKE_Fortran_COMPILER:FILEPATH=$MPI_DIR/bin/mpif77 \
-D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \
-D CMAKE_BUILD_TYPE:STRING=RELEASE \
-D BUILD_SHARED_LIBS:BOOL=ON \
-D CMAKE_MACOSX_RPATH:BOOL=ON \
-D CMAKE_INSTALL_NAME_DIR:PATH=$INSTALL_DIR \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
\
-D TPL_ENABLE_MPI:BOOL=ON \
-D Trilinos_ENABLE_OpenMP:BOOL=ON \
-D TPL_ENABLE_Pthread:BOOL=OFF \
\
-D Trilinos_DISABLE_ENABLED_FORWARD_DEP_PACKAGES=ON \
-D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
-D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
\
-D Trilinos_ENABLE_Teuchos:BOOL=ON \
-D Trilinos_ENABLE_Tpetra:BOOL=ON \
\
-D Trilinos_ENABLE_Kokkos:BOOL=ON \
-D Trilinos_ENABLE_KokkosCore:BOOL=ON \
-D Trilinos_ENABLE_KokkosAlgorithms:BOOL=ON \
-D Kokkos_ENABLE_Serial:BOOL=ON \
-D Kokkos_ENABLE_OpenMP:BOOL=ON \
-D Kokkos_ENABLE_Pthread:BOOL=OFF \
\
-D Trilinos_ENABLE_Zoltan2:BOOL=ON \
-D Zoltan2_ENABLE_Experimental:BOOL=ON\
-D Zoltan2_ENABLE_TESTS:BOOL=ON \
\
-D Trilinos_ENABLE_Xpetra:BOOL=ON \
-D Trilinos_ENABLE_Anasazi:BOOL=ON \
-D Trilinos_ENABLE_Epetra:BOOL=ON \
-D Trilinos_ENABLE_Stratimikos:BOOL=ON \
-D Trilinos_ENABLE_Belos:BOOL=ON \
-D Belos_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_Ifpack2:BOOL=ON \
-D Ifpack2_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_Amesos2:BOOL=ON \
-D Amesos2_ENABLE_TESTS:BOOL=ON \
-D Trilinos_ENABLE_EpetraExt:BOOL=ON \
-D Trilinos_ENABLE_MueLu:BOOL=ON \
-D MueLu_ENABLE_TESTS:BOOL=OFF \
-D MueLu_ENABLE_EXAMPLES:STRING=OFF \
-D MueLu_ENABLE_Epetra:BOOL=OFF \
-D Trilinos_ENABLE_Thyra:BOOL=ON \
-D Trilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON \
-D Trilinos_ENABLE_Teko:BOOL=ON \
-D Teko_ENABLE_TESTS:BOOL=ON \
\
-D Xpetra_Epetra_NO_32BIT_GLOBAL_INDICES=ON \
-D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
-D Tpetra_INST_INT_LONG_LONG:BOOL=ON \
-D Tpetra_INST_INT_INT:BOOL=OFF \
-D Tpetra_INST_DOUBLE:BOOL=ON \
-D Tpetra_INST_FLOAT:BOOL=OFF \
-D Tpetra_INST_COMPLEX_FLOAT:BOOL=OFF \
-D Tpetra_INST_COMPLEX_DOUBLE:BOOL=OFF \
-D Tpetra_INST_INT_LONG:BOOL=OFF \
-D Tpetra_INST_INT_UNSIGNED:BOOL=OFF \
-D Tpetra_INST_SERIAL:BOOL=ON \
-D Tpetra_ENABLE_DEPRECATED_CODE=OFF \
\
\
$SRC_DIR
```

 4.)  After running the above script (configuring), run:
```
    >> make -j4
```

 5.) After compilation is completed, run:
```
    >> make install
```

#### NetCDF + HDF5 (required to use .nc and .g files)
HDF5 needs to be installed before NetCDF.

#####HDF5 instructions:

 1.) Get HDF5 source code at https://www.hdfgroup.org/downloads/hdf5/source-code/ , but be
**SURE** to get the **NON-CMake** version ending in .tar.gz as the .zip has wrong line endings (which will break your build).

 2.) Run the following script (in-source, with your changes to `MPI_DIR` and `INSTALL_DIR`):
```
#!/bin/bash
MPI_DIR=/usr
INSTALL_DIR=/your/install/location
CC=$MPI_DIR/bin/mpicc ./configure --enable-parallel --prefix=$INSTALL_DIR
```

 3.)  After running the above script (configuring), run:
```
    >> make -j4
```

 4.) After compilation is completed, run:
```
    >> make install
```


#####NetCDF instructions:

 1.) Get source code at https://www.unidata.ucar.edu/downloads/netcdf/
     get .tar.gz file under "NetCDF-C Releases"

 2.) Run the following script (with your changes to `HDF5_DIR`, `MPI_DIR`, and `INSTALL_PREFIX`):
```
#!/bin/bash
INSTALL_PREFIX=/your/netcdf
HDF5_DIR=/your/hdf5
MPI_DIR=/usr
CC=$MPI_DIR/bin/mpicc CPPFLAGS=-I${HDF5_DIR}/include LDFLAGS=-L${HDF5_DIR}/lib \
./configure --enable-parallel-tests --prefix=$INSTALL_PREFIX --disable-dap
```

 `--disable-dap` is so that you don't need curl

 3.)  After running the above script (configuring), run:
```
    >> make -j4
```

 4.) After compilation is completed, run:
```
    >> make install
```

 5.) After installation (optional):
```
    >> make check
```

#### Compadre Harness

Now that all of the third party libraries have been installed (HDF5, NetCDF, and Trilinos), it is time to install the Compadre Harness.

 1.) Get the source code at https://github.com/SNLComputation/compadre
     Clone the repo, then check out the **harness** branch:
```
>> git clone https://github.com/SNLComputation/compadre.git
>> cd compadre
>> git fetch origin
>> git checkout -b harness origin/harness
```

 2.)  After cloning, go into compadre directory followed by:
```
    >> mkdir build
    >> cd build
```

 3.) then run this script (with your changes to `MPI_DIR`, and `INSTALL_DIR`):

```
#!/bin/bash

INSTALL_DIR=/your/harness/install
MPI_DIR=/usr
TRILINOS_DIR=/your/trilinos/install
NETCDF_DIR=/your/netcdf/install

# following lines for build directory cleanup
find . ! -name '*.sh' -type f -exec rm -f {} +
find . -mindepth 1 -type d -exec rm -rf {} +

cmake \
    -D CMAKE_CXX_COMPILER=$MPI_DIR/bin/mpic++ \
    -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -D CompadreHarness_USE_Trilinos_CXX_Flags:BOOL=ON \
    -D CompadreHarness_USE_Trilinos_Solvers:BOOL=ON \
    -D CompadreHarness_USE_Netcdf:BOOL=ON \
    -D CompadreHarness_EXAMPLES:BOOL=ON \
    -D CompadreHarness_TESTS:BOOL=ON \
    -D Compadre_EXAMPLES:BOOL=ON \
    -D Compadre_TESTS:BOOL=ON \
    -D Trilinos_PREFIX:FILEPATH=$TRILINOS_DIR \
    -D Netcdf_PREFIX:FILEPATH=$NETCDF_DIR \
    \
    ..
```
Other install scripts are available in the `./scripts` directory of the repo.

 4.)  After running the above script (configuring), run:
```
    >> make -j4
```

 5.) After compilation is completed, run:
```
    >> make install
```

 6.) Go to the `INSTALL_DIR` and run:
```
    >> ctest
```





