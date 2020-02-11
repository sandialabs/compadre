# get source code at https://www.unidata.ucar.edu/downloads/netcdf/
# get .tar.gz file under "NetCDF-C Releases"

HDF5_ROOT=/your/hdf5
INSTALL_PREFIX=/your/netcdf
MPI_DIR=/usr

CC=$MPI_DIR/bin/mpicc CPPFLAGS=-I${HDF5_ROOT}/include LDFLAGS=-L${HDF5_ROOT}/lib \
./configure --enable-parallel-tests --prefix=$INSTALL_PREFIX --disable-dap

# --disable-dap is so that you don't need curl
# follow up with 
# >> make -j4
# >> make install
# (optional) >> make check

