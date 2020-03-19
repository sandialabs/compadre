import re
import sys
import os

# this will be called by build.sh

python_exe = os.getenv("PYTHON")

with open ('cmake_opts.txt.in', 'r' ) as f:
    content = f.read()
    content_new = re.sub('PYTHONEXE', r'%s'%python_exe, content, flags = re.M)
    print(content_new)

if os.getenv("OSX_ARCH"):
    content_new=content_new+"BLAS_LIBRARY_DIRS=/System/Library/Frameworks/Accelerate.framework//Versions/A/Frameworks/vecLib.framework/Versions/A/\n"
    content_new=content_new+"LAPACK_LIBRARY_DIRS=/System/Library/Frameworks/Accelerate.framework//Versions/A/Frameworks/vecLib.framework/Versions/A/\n"

with open ('../cmake_opts.txt', 'w' ) as f:
    f.write(content_new)
