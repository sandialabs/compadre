import re
import sys

assert len(sys.argv) > 1, "Input argument for version missing."
if (len(sys.argv) > 1):
    version  = str(sys.argv[1])

with open ('../cmake/Compadre_Version.txt', 'w' ) as f:
    f.write(version)

