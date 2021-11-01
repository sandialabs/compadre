>> pip install sphinx
>> pip install sphinx_rtd_theme

Be sure to have most recent pycompadre module installed
>> pip uninstall pycompadre
>> python setup.py install

Go to ../doc and run ./build_docs
go here and run `make clean` then `make html`
mv build/html/* to ../doc/html/pycompadre/

If javascript isn't working, double check existence of .nojekyll file in folder
