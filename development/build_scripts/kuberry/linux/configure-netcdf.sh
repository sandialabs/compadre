#CC=mpicc ./configure --enable-netcdf-4 -with-gnu-ld --enable-logging \
#--prefix=$HOME/releases/netcdf \
#--enable-parallel-tests \
#LDFLAGS='LIBS=-lhdf5 -lz -ldl -lmpi'
CC=mpicc CPPFLAGS=-I${HDF5_ROOT}/include LDFLAGS=-L${HDF5_ROOT}/lib \
./configure --enable-parallel-tests --prefix=/home/pakuber/releases/netcdf

#CC=mpicc CPPFLAGS=-I${HDF5_ROOT}/include LDFLAGS=-L${HDF5_ROOT}/lib \
#./configure --disable-shared --enable-parallel-tests --prefix=/home/pakuber/releases/netcdf
