cd python
$PYTHON insert_version.py $PKG_VERSION
cd ..
$PYTHON setup.py install --cmake-file="$RECIPE_DIR/cmake_opts_cpu.txt"
