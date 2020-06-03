#!/bin/bash
DEFAULT_VERSION="1.0.23"

# if looking to install the package, just run 'pip install .' from the root directory of the repo.

# for creating a pypi package, run:
# >> python setup.py sdist 
# >> twine upload dist/*

#
# This file handles installation and packaging of the python interface for the Compadre Toolkit
# Execute this file with a -i or --install flag and it will build and install this package
#

for i in "$@"
do
case $i in
    -v=*|--version=*)
    VERSION="${i#*=}"
    shift # passed argument=value
    ;;
    -e=*|--executable=*)
    EXECUTABLE="${i#*=}"
    shift # passed argument=value
    ;;
    -c|--clean)
    CLEAN=YES
    shift # passed argument with no value
    ;;
    -C|--conda)
    CONDA=YES
    shift # passed argument with no value
    ;;
    -h|--help)
    HELP=YES
    shift # passed argument with no value
    ;;
    *)
    # unknown option
    ;;
esac
done

# install the python package of compadre for the user
if [ "$HELP" == "YES" ]; then

    echo "-e  | --executable  Python executable to be used for package installation"
    echo "-c  | --clean       Removes temporary files created by packaging and installation"
    echo "-h  | --help        See the help  screen that you are currently reading"
    echo "-v= | --version=    (OPTIONAL UNLESS PACKAGING)"
    exit 0

fi

# remove temporary files created by install and packaging
if [ "$CLEAN" == "YES" ]; then
    rm -r ../dist
    rm -r ../pycompadre.egg-info
    exit 0
fi

# output variables used
echo "VERSION  = ${VERSION}"
echo "CONDA: ${CONDA}"
if [ "$EXECUTABLE" == "" ]; then
    EXECUTABLE=`which python`
    echo "$0: No Python executable provided with \" -e=*\", so first python found in search path is used: $EXECUTABLE"
fi
echo "PYTHON EXECUTABLE: ${EXECUTABLE}"

# handle version argument for packaging and install
if [ "$VERSION" == "" ] && [ "$CONDA" == "YES" ]; then
    echo "$0: A version number is required to be provided for packaging. \" -v=*\""
    exit 1
fi
if [ "$VERSION" == "" ]; then
    echo "$0: A version number was not provided, so version number set to DEFAULT_VERSION of $DEFAULT_VERSION"
    VERSION="$DEFAULT_VERSION"
fi

# (NOT for users) create a conda package
if [ "$CONDA" == "YES" ]; then

    # conda activate builder
    # conda build purge-all
    # conda-bld should be cleared as well 

    
    rm -rf ../build
    rm -rf ../meta.yaml
    rm -rf ../cmake_opts.txt

    $EXECUTABLE insert_version.py $VERSION
    echo "version $VERSION inserted into setup file."

    #cp update_conda_cmake.py ../update_conda_cmake.py
    cp meta.yaml.in ../meta.yaml
    cp build.sh ..
    #cp conda_build_config.yaml.in ../conda_build_config.yaml
    cd ..
    conda-build . --python=3.7
    # --python=3.6 --python=3.7
    cd pycompadre

fi

# instructions for updating conda package
# 1.) git fetch origin on compadre repo
# 2.) from the root of the repo, with master checked out, run >> git merge origin/conda_files --allow-unrelated-histories
# 3.) >> conda-build . --python=3.6 --python=3.7 --python=3.8
# 4.) >> anaconda login
# 5.) >> anaconda upload /some/path/to/compadre.tar.bz2
