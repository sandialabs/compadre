# Installing the Compadre Harness
 
## Compadre Harness Installation Instructions

### Requirements

  **Compadre Harness includes the Compadre Toolkit as a subdirectory, which will be configured and built while installing the Harness.

  **All build instructions are only tested on linux platforms.

  VTK and NetCDF are the two means by which files can be read into the Harness. It is expected that a user will have BOTH installed, but each can be individually disable in the CMake configure script using the variable

```
-D CompadreHarness_USE_VTK:BOOL=OFF
```

or 

```
-D CompadreHarness_USE_Netcdf:BOOL=OFF
```

#### Trilinos

  The Compadre Harness requires Trilinos with a certain set of packages in Trilinos turned on. Example scripts are provided which turn on the needed packages in the Trilinos installation section.


 1.)  Get Trilinos source at: https://github.com/trilinos/Trilinos 

 2.)  After cloning, go into Trilinos directory followed by:
```
    >> mkdir build
    >> cd build
```
 3.) then run this script:

```
#!/bin/bash

SRC_DIR=$HOME/code/Trilinos
BUILD_DIR=$HOME/code/Trilinos/build
INSTALL_DIR=$BUILD_DIR/install

MPI_DIR=/usr

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
\
-D TPL_ENABLE_Netcdf:BOOL=ON \
-D Netcdf_INCLUDE_DIRS:FILEPATH=$NETCDF_DIR/include \
-D Netcdf_LIBRARY_DIRS:FILEPATH=$NETCDF_DIR/lib \
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

#### NetCDF + HDF5
HDF5 needs to be installed before NetCDF.

HDF5 instructions:

NetCDF instructions:

#### VTK

  VTK and NetCDF are the two means by which files can be read into the Harness.




  1.) Copy one of the examples bash script files from ./scripts to the folder where you would like to build the project.
      Ideally, this should not be at the root of the repository as it is never suggested to build in-source.
  
  example:

```
  >> mkdir build
  >> cp ./scripts/script_you_choose.sh build
```
  
  2.) Edit the script file that you copied to your soon-to-be build folder.
      Make changes to these files to suit your needs (KokkosCore_PREFIX, etc...)
      Then, run the modified bash script file.
  
  (assumes you moved the script to ./build as in #1 )
  
```
  >> cd ./build
  >> vi script_you_choose.sh
```
  (make any changes and save, [Python Interface and Matlab examples](Python-Interface-Install.md))
  
```
  >> ./script_you_choose.sh
```
      
  3.) Build the project.
  
```
  >> make -j4                      # if you want to build using 4 processors
  >> make install
```
  
  4.) Test the built project by exercising the suite of tests.
  
```
  >> ctest
```

  5.) Build doxygen documentation for the project by executing

```
  >> make Doxygen
  >> [your favorite internet browser executable] doc/output/html/index.html
```

   
  If some tests fail, be sure to check the error as it is possible that you have not configured CMake
  as to where it should locate libraries like Kokkos, Python, etc...
  If a library is missing but not turned on in the CMake options, then the test will simply fail.
  
## Importing Project Into Eclipse

From https://stackoverflow.com/questions/11645575/importing-a-cmake-project-into-eclipse-cdt,
the instructions for importing from CMake into eclipse are as follows:

> First, choose a directory for the CMake files. I prefer to keep my Eclipse workspaces in 
> ~/workspaces and the source code in ~/src. Data which I need to build or test the project 
> goes in subdirs of the project's workspace dir, so I suggest doing the same for CMake.
> 
> Assuming both your workspace and source folders are named someproject, do:
> 
> cd ~/workspaces/someproject
> mkdir cmake
> cd cmake
> cmake -G "Eclipse CDT4 - Unix Makefiles" ~/src/someproject
> 
> Then, in your Eclipse workspace, do:
> 
> File > Import... > General > Existing Projects into Workspace
> 
> Check Select root directory and choose ~/workspaces/someproject/cmake. Make sure Copy projects into workspace is NOT checked.
> 
> Click Finish and you have a CMake project in your workspace.
> 
> Two things to note:
> 
>   I used cmake for the workspace subdir, but you can use a name of your choice.
>   If you make any changes to your build configuration (such as editing Makefile.am), you will need to re-run the 
>   last command in order for Eclipse to pick up the changes.

