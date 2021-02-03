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
import distutils

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from setuptools.command.install import install

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print("build_extension called.")
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Parse string for kokkos architecture

        cmake_config = os.getenv("CMAKE_CONFIG_FILE")
        if cmake_config:
            print("Custom cmake args file set to: %s"%(cmake_config,))
        else:
            cmake_config = ""
            print("Custom cmake args file not set.")

        #cmake_file_string = ""
        #try:
        #    if cmake_file != None:
        #        cmake_file_string = str(cmake_file)
        #        print("Custom cmake args file set to: %s"%(cmake_file_string,))
        #    else:
        #        print("Custom cmake args file not set.")
        #except:
        #    print("Custom cmake args file not set.")

        # Configure CMake
        #config = 'Debug' if self.debug else 'Release'
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DCMAKE_CXX_FLAGS= -O3 ',
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DGMLS_Module_DEST=' + extdir,
                      '-DCMAKE_INSTALL_PREFIX=' + extdir,
                      '-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9',
                      '-DCompadre_USE_PYTHON:BOOL=ON',
                      '-DCompadre_USE_MATLAB:BOOL=ON',
                      '-DCompadre_EXAMPLES:BOOL=OFF',
                      '-DPYTHON_CALLING_BUILD:BOOL=ON',]

        cmake_file_list = list()
        #if (cmake_file_string != ""):
        if (cmake_config != ""):
            #cmake_arg_list = [line.rstrip('\n') for line in open(cmake_file_string)]
            cmake_arg_list = [line.rstrip('\n') for line in open(cmake_config)]
            for arg in cmake_arg_list:
                cmake_args.append('-D'+arg)
            print('Custom CMake Args: ', cmake_arg_list)

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("cmake/Compadre_Version.txt", "r") as fh:
    version_string = fh.read()

setup(
    name='pycompadre',
    version=version_string,
    author='Paul Kuberry',
    author_email='pkuberry@gmail.com',
    description="Compatible Particle Discretization and Remap",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SNLComputation/compadre",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    install_requires=['cmake>=3.10.0',],
    ext_modules=[CMakeExtension('pycompadre'),],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    tests_require=['nose','numpy'],
    test_suite='nose.collector',
    include_package_data=True,
    zip_safe=False,
)
