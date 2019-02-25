# Installing the Python interface and/or the Matlab examples which use the Python interface
Modifying the build script, for instance "script_you_choose.sh", we change or add the following CMake variables:
```
-D Compadre_USE_PYTHON:BOOL=ON \
```
to enable the Python interface to the toolkit and optionally,
```
-D Compadre_USE_MATLAB:BOOL=ON \
``` 
to enable to the Matlab examples.

Python must be enabled in order to use the Matlab tests and install scripts. There is no interface from Matlab directly to the C++ code. This only takes place through the Python interface.


The CMake build system will attempt to determine which Python executable you are using. If you would like to specify a particular version of Python, set the CMake variable:
```
-D PYTHON_EXECUTABLE:FILEPATH='path/to/your/python' \
``` 
Otherwise, `which python` will be called and whatever is returned will be used.


**Notes:**

Numpy header files are needed in order to build the interface to Python. The PYTHON_EXECUTABLE that you set will be used to attempt to **automatically determine** the location of these files. However, it is possible that this search will fail, in which case y
ou will need to provide the additional CMake variable:

```
-D Numpy_INCLUDE_DIRS:PATH='path/to/your/numpy/' \
``` 
As an example, the build system will search for /some/path/to/some/files/ending/in/numpy/arrayobject.h and chop off the ending numpy/arrayobject.h, so in this example you would set the variable to:
```
-D Numpy_INCLUDE_DIRS:PATH='/some/path/to/some/files/ending/in/' \
``` 



The Matlab examples require at least Matlab version 2017a.

Back to [Installation of Compadre Toolkit](Compadre-Install.md)
