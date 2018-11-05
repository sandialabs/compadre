rm -rf GMLS_Module_wrap.cxx
swig -python -c++ GMLS_Module.i
cat python_lib_load_script.txt GMLS_Module.py > GMLS_Module.py.in;
rm -rf GMLS_Module.py
