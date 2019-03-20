python insert_version.py $1
echo "version $1 inserted into setup file."

rm -rf ../dist
cp pyproject.toml ..

# step down to root of repo
cd ..

python setup.py bdist_wheel sdist
echo "bdist_wheel and sdist complete."

rm -rf build/ && rm -rf python_src/CMakeFiles && rm -rf python_src/CMakeCache.txt && cd dist && tar -xzvf compadre-$1.tar.gz && cd compadre-$1 && cp PKG-INFO ../.. && cd ../.. && rm -rf dist 
echo "files copied from dist."

tar -czvf compadre-$1.tar.gz *
echo "files tar balled."

mkdir dist && mv compadre-$1.tar.gz dist
echo "tar ball moved to dist."

# test pypi site
#python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose
#echo "uploaded to test pypi"

# real pypi site
twine upload dist/*
echo "uploaded to real pypi"

# step back up to calling folder
cd python

# command to install from test pypi
#python -m pip install --index-url https://test.pypi.org/simple/ --no-deps --upgrade compadre

