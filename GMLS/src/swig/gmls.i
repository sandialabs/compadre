%{
#define SWIG_FILE_WITH_INIT
%}
/*%include numpy.i*/
%include std_string.i
%init %{
import_array();
%}


%module gmls
%{
#include "GMLS_Python.hpp"
#include "GMLS_Kokkos.hpp"
#include "Kokkos_Core.hpp"
#include "Python.h"
#include "numpy/arrayobject.h"
%}

/* Parse the header file to generate wrappers */
%include "GMLS_Python.hpp"
