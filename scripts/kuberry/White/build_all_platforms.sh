mkdir build-kepler35
cp scripts/kuberry/White/my-do-configure-rhel7ft.sh build-kepler35
cd build-kepler35
./my-do-configure-rhel7ft.sh
make -j
cd ..

mkdir build-pascal60
cp scripts/kuberry/White/my-do-configure-rhel7g.sh build-pascal60
cd build-pascal60
./my-do-configure-rhel7g.sh
make -j
cd ..

mkdir build-cpu
cp scripts/kuberry/White/my-do-configure-rhel7anycpu.sh build-cpu
cd build-cpu
./my-do-configure-rhel7anycpu.sh
make -j
cd ..
