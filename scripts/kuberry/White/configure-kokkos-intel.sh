# INTEL
make clean
../generate_makefile.bash --arch=SNB,Kepler35 --with-cuda-options="enable_lambda" --cxxflags="-fPIC" --with-cuda=/home/projects/ppc64le-pwr8-nvidia/cuda/9.2.88 --with-openmp --compiler=$HOME/releases/kokkos/bin/nvcc_wrapper --prefix=$HOME/releases/kokkos/install/kepler35
make -j install

make clean
../generate_makefile.bash --arch=SNB,Pascal60 --with-cuda-options="enable_lambda" --cxxflags="-fPIC" --with-cuda=/home/projects/ppc64le-pwr8-nvidia/cuda/9.2.88 --with-openmp --compiler=$HOME/releases/kokkos/bin/nvcc_wrapper --prefix=$HOME/releases/kokkos/install/pascal60
make -j install

make clean
../generate_makefile.bash --arch=SNB --with-cuda-options="enable_lambda" --cxxflags="-fPIC" --with-cuda=/home/projects/ppc64le-pwr8-nvidia/cuda/9.2.88 --with-openmp --compiler=$HOME/releases/kokkos/bin/nvcc_wrapper --prefix=$HOME/releases/kokkos/install/cpu
make -j install
