#!/bin/bash

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
    -e=*|--executable=*)
    EXECUTABLE="${i#*=}"
    shift # passed argument=value
    ;;
    -c|--clean)
    CLEAN=YES
    shift # passed argument with no value
    ;;
    -p|--package)
    PACKAGE=YES
    shift # passed argument with no value
    ;;
    -P|--PACKAGE)
    PACKAGE=YES
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
    echo "-p  | --package     Create a python package that can be uploaded to Pypi"
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
if [ "$EXECUTABLE" == "" ]; then
    EXECUTABLE=`which python`
    echo "$0: No Python executable provided with \" -e=*\", so first python found in search path is used: $EXECUTABLE"
fi
echo "PYTHON EXECUTABLE: ${EXECUTABLE}"

# (NOT for users) create a python package that can be uploaded to pypi 
if [ "$PACKAGE" == "YES" ]; then


    rm -rf ../dist
    rm -rf ../build
    rm -rf ../build.sh
    rm -rf ../pycompadre.egg-info
    rm -rf ../pycompadre-serial.egg-info
    cp cmake_opts_perf.txt ../cmake_opts.txt
    cp MANIFEST.in ../MANIFEST.in

    cd ..

    # requires pip package 'build'
    CMAKE_CONFIG_FILE=cmake_opts.txt $EXECUTABLE -m build --sdist

    rm -rf pycompadre.egg-info
    echo "sdist complete."

    cd pycompadre
    # follow up with twine upload ../dist/*

fi
