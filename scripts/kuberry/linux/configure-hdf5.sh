# source code at https://www.hdfgroup.org/downloads/hdf5/source-code/
# be SURE to get the NON-cmake version ending in .tar.gz 
# (.zip has wrong line endings)

INSTALL_PREFIX=/your/favorite/install/location

CC=mpicc ./configure --enable-parallel --prefix=$INSTALL_PREFIX

# follow with 
# >> make -j4 
# >> make install
