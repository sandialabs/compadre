# Installing Kokkos
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

