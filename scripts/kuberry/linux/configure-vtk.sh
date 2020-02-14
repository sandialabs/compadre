# get source code from https://vtk.org/download/
# under "Latest Release" and "Source" get the .tar.gz file

# after untarring, go into directory 
# >> mkdir build
# >> cd build
# inside of build, run this script

rm -rf CMakeCache* CMakeFiles*

INSTALL_PREFIX=/home/paul/code/VTK
MPI_DIR=/usr

cmake \
-D BUILD_SHARED_LIBS:BOOL=ON \
-D CMAKE_INSTALL_PREFIX:PATH=$INSTALL_PREFIX \
-D CMAKE_INSTALL_NAME_DIR:STRING=$INSTALL_PREFIX/lib \
-D CMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON \
-D BUILD_TESTING:BOOL=OFF \
-D CMAKE_C_COMPILER=$MPI_DIR/bin/mpicc \
-D CMAKE_CXX_COMPILER=$MPI_DIR/bin/mpic++ \
-D CMAKE_INSTALL_RPATH:STRING=$INSTALL_PREFIX/lib \
-D VTK_SMP_IMPLEMENTATION_TYPE=OPENMP \
-D Module_vtkFiltersParallelMPI:BOOL=ON \
-D Module_vtkIOMPIParallel:BOOL=ON \
-D Module_vtkIOParallelExodus:BOOL=ON \
-D Module_vtkIOParallelNetCDF:BOOL=ON \
-D Module_vtkParallelMPI:BOOL=ON \
-D Module_vtkRenderingParallel:BOOL=ON \
-D MPIEXEC_MAX_NUMPROCS:STRING=8 \
-D Module_vtkIOMPIImage:BOOL=ON \
-D VTK_Group_MPI:BOOL=ON \
-D VTK_MPI_MAX_NUMPROCS:STRING=8 \
-D CMAKE_BUILD_TYPE:STRING=Release \
 ..

# after running this script, run
# >> make -j4
# >> make install

# while running this script, may encounter needing X11_Xt_LIB
# on Ubuntu, >> sudo apt-get install libxt-dev
# on RHEL,   >> sudo yum install libXt-devel 
