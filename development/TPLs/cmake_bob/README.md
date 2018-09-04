# cmake_bob
> modern CMake helpers

`bob.cmake` contains CMake functions and macros to help you quickly set up
a modern [CMake package][0] which properly exports its targets and upstream
dependencies so that downstream use is easy.
It is mainly geared towards HPC users who tend to install packages
in non-system directories.

## Installing / Getting started

A quick introduction of the minimal setup you need to get a hello world up &
running.
To use `bob.cmake`, simply copy the file somewhere into your project
(for example, into a `cmake/` subdirectory), then include it in
your top-level `CMakeLists.txt` file and at minimum call
`bob_begin_package` and `bob_end_package`:

```cmake
project(...)
...
include(cmake/bob.cmake)
...
bob_begin_package()
...
bob_end_package()
```

The call to `bob_begin_package` will set up `RPATH` and testing options,
and the call to `bob_end_package` will export all targets and upstream
package depencies automatically.

## Features

The main tasks of `bob.cmake` are:
* Install and export your targets.
* Allow users of your package to automatically load upstream
package information as well.
* Setup a [full RPATH][1] for your libraries and executables.
This makes them equally easy to use when not installed in system
paths, without the need for `PATH` and `LD_LIBRARY_PATH` editing.
* Automatically generate options to toggle upstream dependencies
and indicate where they are installed.

## Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.

## Licensing

The code in this project is licensed under MIT license.

[0]: https://cmake.org/cmake/help/v3.2/manual/cmake-packages.7.html
[1]: https://cmake.org/Wiki/CMake_RPATH_handling#Always_full_RPATH
