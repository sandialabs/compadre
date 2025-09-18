# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525, the U.S. Government retains certain rights 
# in this software.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of NTESS or the U.S. Government.
# 
# 
#    Questions? Contact Paul Kuberry  (pakuber@sandia.gov),
#                       Peter Bosler  (pabosle@sandia.gov), or
#                       Nat Trask     (natrask@sandia.gov)
# 
# 
# 
# Copyright (c) 2016 The Pybind Development Team, All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to the author of this software, without
# imposing a separate written license agreement for such Enhancements, then you
# hereby grant the following license: a non-exclusive, royalty-free perpetual
# license to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such enhancements or
# derivative works thereof, in binary and source code form.
#
# This file is a modification of the setup.py found at:
# https://github.com/pybind/cmake_example/blob/master/setup.py
#
# Significant portions of CMakeBuild::build_extension have been added, as well as setup.

import os
import re
import sys
import platform
import subprocess
from packaging.version import Version
import shutil

from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
import importlib.resources
import platform


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        if not self.sourcedir.endswith(os.path.sep):
            self.sourcedir += os.path.sep

# Overload Command for SetupTest to prevent build_ext being called again
class CustomTest(Command):
    '''
    Tests the installed pycompadre module
    '''
    user_options=list()
    def initialize_options(self): pass
    def finalize_options(self): pass
    def run(self):
        try:
            import pycompadre
        except:
            print('Install pycompadre before running test')
        pycompadre.test()

class CustomBuild(build_ext):

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except oserror:
            raise runtimeerror("cmake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions) + 
                               "\ninstall cmake with `pip install cmake` or install from: https://cmake.org/download/.")

        cmake_version = Version(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < Version('3.16.0'):
            raise runtimeerror("cmake >= 3.16.0 is required")
        assert sys.version_info >= (3,6), "\n\n\n\n\npycompadre requires python version 3.6+\n\n\n\n\n"

        for ext in self.extensions:
            self.configure_pycompadre(ext)

    def configure_pycompadre(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name))+"/pycompadre")
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_config = os.getenv("CMAKE_CONFIG_FILE")
        if cmake_config:
            #  check if absolute or relative path
            if os.path.isabs(cmake_config):
                print("Custom cmake args file set to: %s"%(cmake_config,))
            else:
                if sys.path[0]=='' or not os.path.isdir(sys.path[0]):
                    assert False, "CMAKE_CONFIG_FILE (%s) must be given as an absolute path when called from pip."
                else:
                    print("CMAKE_CONFIG_FILE given as a relative path: %s"%(cmake_config,))
                    cmake_config = sys.path[0] + os.sep + cmake_config
                    print("Custom cmake args file set to absolute path: %s"%(cmake_config,))
            if not os.path.exists(cmake_config):
                assert False, "CMAKE_CONFIG_FILE specified, but does not exist."
        else:
            # look for cmake_opts.txt and use it if found
            if os.path.exists(ext.sourcedir+"cmake_opts.txt"):
                cmake_config = ext.sourcedir+"cmake_opts.txt"
                print("CMAKE_CONFIG_FILE not set, but cmake_opts.txt FOUND, so using: %s"%(cmake_config,))
            else:
                cmake_config = ""
                print("Custom cmake args file not set.")

        if Version(platform.python_version()) >= Version('3.9.0'):
            pybind11_path = str(importlib.resources.files('pybind11'))
        else:
            import pybind11
            pybind11_path = str(os.path.abspath(os.path.dirname(pybind11.__file__)))

        # Configure CMake
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DCMAKE_INSTALL_PREFIX=' + extdir,
                      '-DCompadre_USE_PYTHON:BOOL=ON',
                      '-Dpybind11_DIR=' + pybind11_path + '/share/cmake/pybind11/',
                      '-DPYTHON_CALLING_BUILD:BOOL=ON',]

        cmake_file_list = list()
        if (cmake_config != ""):
            cmake_arg_list = [line.rstrip('\n') for line in open(cmake_config)]
            for arg in cmake_arg_list:
                # handle non-support of '-march=native' on arm with apple-clang
                if platform.processor()=="arm" and "-march=native" in arg:
                    arg = arg.replace("-march=native", "")
                    print("Warning: -march=native incompatible with arm so flag was removed")
                cmake_args.append('-D'+arg)
            print('Provided Custom CMake Args: ', cmake_arg_list)

        # turn off examples and tests unless specified in a file
        tests_in_cmake_args = False
        for arg in cmake_args:
            if 'Compadre_TESTS' in arg:
                tests_in_cmake_args = True
                break
        if not tests_in_cmake_args:
            cmake_args += ['-DCompadre_TESTS:BOOL=OFF']
        examples_in_cmake_args = False
        for arg in cmake_args:
            if 'Compadre_EXAMPLES' in arg:
                examples_in_cmake_args = True
                break
        if not examples_in_cmake_args:
            cmake_args += ['-DCompadre_EXAMPLES:BOOL=OFF']

        # add level 3 optimization if not specified in a file
        flag_in_cmake_args = False
        for arg in cmake_args:
            if 'CMAKE_CXX_FLAGS' in arg:
                flag_in_cmake_args = True
                break
        if not flag_in_cmake_args:
            cmake_args += ['-DCMAKE_CXX_FLAGS= -O3 -g ']

        # add level 3 optimization if not specified in a file
        build_type_in_cmake_args = False
        for arg in cmake_args:
            if 'CMAKE_BUILD_TYPE' in arg:
                build_type_in_cmake_args = True
                break

        # allow 'python setup.py build_ext --debug install' to have precedence over CMAKE_BUILD_TYPE
        if self.debug and not build_type_in_cmake_args:
            cfg = 'Debug'
            cmake_args += ['-DCMAKE_BUILD_TYPE=Debug']
        elif not self.debug and not build_type_in_cmake_args:
            cfg = 'None'
            cmake_args += ['-DCMAKE_BUILD_TYPE=None']
        elif not self.debug: # was specified in file only
            for arg in cmake_args[:]:
                if 'CMAKE_BUILD_TYPE' in arg:
                    # break down and copy to cfg
                    cfg = arg.split('=')[1]
                    break
        else: # was specified in file and --debug in command
            # delete specified from cmake_args 
            for arg in cmake_args[:]:
                if 'CMAKE_BUILD_TYPE' in arg:
                    cmake_args.remove(arg)
            cmake_args += ['-DCMAKE_BUILD_TYPE=Debug']
            cfg = 'Debug'

        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print("CMake Args:", cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)

        # copy __init__.py to install directory
        shutil.copyfile(self.build_temp + "/__init__.py", extdir + "/__init__.py")
        # move examples/* from source directory to install directory (supports pycompadre.test())
        try:
            os.mkdir(extdir + "/examples")
        except:
            pass
        files_to_copy = os.listdir(ext.sourcedir + "pycompadre/examples/")
        for fname in files_to_copy:
            try:
                shutil.copyfile(ext.sourcedir + "pycompadre/examples/" + fname, 
                                extdir + "examples/" + os.path.basename(fname))
            except:
                print("WARNING: Copying pycompadre test files failed.")

        print("Build Args:", build_args)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("cmake/Compadre_Version.txt", "r") as fh:
    version_string = str(Version(fh.read()))

setup(
    cmdclass={
        'build_ext' : CustomBuild,
        'test': CustomTest,
    },
    version=version_string,
    description = "Compatible Particle Discretization and Remap",
    long_description_content_type = "text/markdown",
    long_description=long_description,
    ext_modules=[CMakeExtension('_pycompadre'),],
)
