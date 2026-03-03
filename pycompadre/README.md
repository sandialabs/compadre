# pyCOMPADRE: COMpatible PArticle Discretization and REmap Toolkit

## Installation

[Installation of pycompadre from pypi](https://github.com/sandialabs/compadre/wiki/Python-Package-(pip-conda))

[Installation of pycompadre from source](https://github.com/sandialabs/compadre/wiki/Installing-from-source)

## Modifying Configuration:

  pycompadre uses a pyproject.toml file in conjunction with setup.py to build
  COMPADRE in C++ and pycompadre (nanobind wrapping C++) with CMake.

  Supported installation patterns look like: 

    >> pip install [source_directory]

  If you would like to modify CMake arguments for custom installations of pycompadre,
  examples can be found in the pycompadre folder with names like cmake_opts*.txt
  These CMake options files can be customized and indicated to the installer using:
  
    >> CMAKE_CONFIG_FILE=/your/location/of/cmake_opts.txt pip install [source_directory]

  or you can get the source from pypi with:

    >> CMAKE_CONFIG_FILE=/your/location/of/cmake_opts.txt pip install pycompadre

  (but be careful not to call that from the root of this repository, or it could be 
   confused with the pycompadre/ folder that has no pyproject.toml or setup.py file)

## Debugging: 

  When pip installing pycompadre, you can define the environment variable 
  CMAKE_BUILD_TYPE={Release,RelWithDebInfo,Debug}. Choosing 'Debug' will enabled
  additional, performance degrading integrity checks that are likely quite useful
  when creating something new with pycompadre. Once you are ready for production,
  you can pip install again specifying CMAKE_BUILD_TYPE=Release.

    >> CMAKE_BUILD_TYPE=Debug CMAKE_CONFIG_FILE=/your/location/of/cmake_opts.txt pip install pycompadre

  Questions, comments, problems to pakuber@sandia.gov

