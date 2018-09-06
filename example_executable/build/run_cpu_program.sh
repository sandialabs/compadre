# make sure that you have first built the program before running this script
mpirun -n 4 ./tpetraCoordsTest.exe --i=../parameters/parameters_amg.xml --kokkos-threads=2
./GMLS_PointValuesTest_Kokkos 4 200 3
./GMLS_PointValuesTest_Kokkos 4 200 2
./GMLS_PointValuesTest_Kokkos 4 200 1
