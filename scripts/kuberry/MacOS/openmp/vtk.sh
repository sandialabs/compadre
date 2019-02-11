rm -rf CMakeCache* CMakeFiles*

VTK_ROOT=/Users/pakuber/Compadre/VTK-8.0.1/build

cmake \
-D BUILD_SHARED_LIBS:BOOL=ON \
-D CMAKE_INSTALL_PREFIX:PATH=$VTK_ROOT/install \
-D CMAKE_INSTALL_NAME_DIR:STRING=$VTK_ROOT/install/lib \
-D CMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON \
-D BUILD_TESTING:BOOL=OFF \
-D CMAKE_C_COMPILER=mpicc \
-D CMAKE_CXX_COMPILER=mpic++ \
-D CMAKE_INSTALL_RPATH:STRING=$VTK_ROOT/install/lib \
-D VTK_SMP_IMPLEMENTATION_TYPE=PTHREADS \
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
-D CMAKE_BUILD_TYPE:STRING=Releases \
 ..
