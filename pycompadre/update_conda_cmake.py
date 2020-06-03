import re
import sys
import os

# this will be called by build.sh

assert len(sys.argv) > 2, "Input arguments for file in and or out is missing."
if (len(sys.argv) > 2):
    fin  = str(sys.argv[1])
    fout  = str(sys.argv[2])

python_exe = os.getenv("PYTHON")

with open (fin, 'r' ) as f:
    content = f.read()
    content_new = re.sub('PYTHONEXE', r'%s'%python_exe, content, flags = re.M)
    print(content_new)

if os.getenv("OSX_ARCH"):
    content_new=content_new+"BLAS_LIBRARY_DIRS=/System/Library/Frameworks/Accelerate.framework//Versions/A/Frameworks/vecLib.framework/Versions/A/\n"
    content_new=content_new+"LAPACK_LIBRARY_DIRS=/System/Library/Frameworks/Accelerate.framework//Versions/A/Frameworks/vecLib.framework/Versions/A/\n"

with open (fout, 'w' ) as f:
    f.write(content_new)
