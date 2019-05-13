#!/bin/bash
DEFAULT_VERSION="1.0.23"

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
    -i|--install)
    INSTALL=YES
    shift # passed argument with no value
    ;;
    -p|--package)
    PACKAGE=YES
    shift # passed argument with no value
    ;;
    -c|--clean)
    CLEAN=YES
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

    echo "-i  | --install     Install the compadre python package"
    echo "-p  | --package     Create a python package that can be uploaded to Pypi"
    echo "-e  | --executable  Python executable to be used for package installation"
    echo "-c  | --clean       Removes temporary files created by packaging and installation"
    echo "-h  | --help        See the help  screen that you are currently reading"
    echo "-v= | --version=    (OPTIONAL UNLESS PACKAGING)"
    exit 0

fi

# remove temporary files created by install and packaging
if [ "$CLEAN" == "YES" ]; then
    rm -r ../dist
    rm -r ../compadre.egg-info
    rm ../setup.py
    rm ../pyproject.toml
    exit 0
fi

# output variables used
echo "VERSION  = ${VERSION}"
echo "INSTALL: ${INSTALL}"
echo "CREATE PACKAGE: ${PACKAGE}"
if [ "$EXECUTABLE" == "" ]; then
    EXECUTABLE=`which python`
    echo "$0: No Python executable provided with \" -e=*\", so first python found in search path is used: $EXECUTABLE"
fi
echo "PYTHON EXECUTABLE: ${EXECUTABLE}"

# handle version argument for packaging and install
if [ "$VERSION" == "" ] && [ "$PACKAGE" == "YES" ]; then
    echo "$0: A version number is required to be provided for packaging. \" -v=*\""
    exit 1
fi
if [ "$VERSION" == "" ]; then
    echo "$0: A version number was not provided, so version number set to DEFAULT_VERSION of $DEFAULT_VERSION"
    VERSION="$DEFAULT_VERSION"
fi

# check that Python is version 3.0 or higher
check_python_version () {
    VERSION_SUFFICIENT=`$1 -c "import sys; print(sys.version_info >= (3,0));"`
    if [ "$VERSION_SUFFICIENT" == "False" ]; then
        echo "Python package requires Python version 3.0 or higher."
        exit 1
    fi
}

# install the python package of compadre for the user
if [ "$INSTALL" == "YES" ]; then

    check_python_version $EXECUTABLE

    python insert_version.py $VERSION
    echo "version $VERSION inserted into setup file."

    rm -rf ../dist
    cp pyproject.toml ..

    cd ..
    $EXECUTABLE setup.py install --force

    cd python

fi

# (NOT for users) create a python package that can be uploaded to pypi 
if [ "$PACKAGE" == "YES" ]; then

    check_python_version $EXECUTABLE

    python insert_version.py $VERSION
    echo "version $VERSION inserted into setup file."

    rm -rf ../dist
    cp pyproject.toml ..

    cd ..

    rm -rf compadre.egg-info
    
    $EXECUTABLE setup.py bdist_wheel sdist
    echo "bdist_wheel and sdist complete."
    
    rm -rf build/ && cd dist && tar -xzvf compadre-$VERSION.tar.gz && cd compadre-$VERSION && cp PKG-INFO ../.. && cd ../.. && rm -rf dist 
    echo "files copied from dist."
    
    tar -czvf compadre-$VERSION.tar.gz *
    echo "files tar balled."
    
    mkdir dist && mv compadre-$VERSION.tar.gz dist
    echo "tar ball moved to dist."

    cd python

fi

# various commands for uploading to pypi

# upload to the test pypi site
#python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose
#echo "uploaded to test pypi"

# upload to the real pypi site
#twine upload dist/*
#echo "uploaded to real pypi"

# command to install from test pypi
#python -m pip install --index-url https://test.pypi.org/simple/ --no-deps --upgrade compadre

