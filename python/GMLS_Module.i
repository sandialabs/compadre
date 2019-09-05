%{
#define SWIG_FILE_WITH_INIT
%}
/*%include numpy.i*/
%include std_string.i
%include exception.i
%init %{
import_array();
%}


%module GMLS_Module
%{
#include "GMLS_Python.hpp"
#include "Compadre_GMLS.hpp"
#include "Kokkos_Core.hpp"
#include "Python.h"
#include "numpy/arrayobject.h"
%}

%exception { 
    try {
        $action
    } catch(std::exception &_e) {
        PyErr_SetString(PyExc_RuntimeError,(&_e)->what());
        SWIG_fail;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown exception");
        SWIG_fail;
    }
}

/* Parse the header file to generate wrappers */
%include "GMLS_Python.hpp"
