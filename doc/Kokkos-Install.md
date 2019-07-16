# Installing Kokkos

There are two paths for providing a Kokkos installation, needed by the toolkit:

# A.) Let Compadre Toolkit configure and build Kokkos

Since the source code for Kokkos is now bundled with the Compadre Toolkit, users can simply leave the CMake variable KokkosCore_PREFIX="" and optionally provide a few configuration details via KokkosCore_FLAGS and KokkosCore_ARCH. 

Valid choices for KokkosCore_ARCH can be found in kokkos/cmake/kokkos_options.cmake and should be semicolon separated.

# or B.) Installing Kokkos from source and providing the location to the toolkit

1.) Download Kokkos from github via:

```
>> git clone https://github.com/kokkos/kokkos.git
```

2.) Build kokkos via:
```
>> cd kokkos
>> mkdir build
>> cd build
```

3.) Create file called configure.sh 
insert the following two lines (for basic cpu with openmp build), but change prefix to whatever you
would like. if you do not change it, then kokkos will be installed in /some/path/to/kokkos/build/install-openmp

```
CXX="/path/to/your_c++_compiler"
../generate_makefile.bash --compiler=$CXX --with-openmp --with-serial --prefix="../install-openmp" --cxxflags="-fPIC"
```

4.) Set permissions on configure.sh so it can be run with:
```
>> chmod u+x configure.sh
```

5.) Run the script to configure.
```
>> ./configure.sh
```

6.) Build make install the project.
```
>> make -j4
>> make install
```

7.) Note where you installed Kokkos, as this install folder will be the location that your Compadre Toolkit build scripts should use as the value for the variable KokkosCore_PREFIX.

