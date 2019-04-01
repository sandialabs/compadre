
# The following code will search 'mm/dd/yyyy' (e.g. NOV/30/2016 ),
# and replace with 'mm-dd-yyyy' in multi-line mode.
import re
import sys

assert len(sys.argv) > 1, "Input argument for version missing."
if (len(sys.argv) > 1):
    version  = str(sys.argv[1])

with open ('setup.py.in', 'r' ) as f:
    content = f.read()
    content_new = re.sub('VERSION', r'%s'%version, content, flags = re.M)
    print(content_new)

with open ('../setup.py', 'w' ) as f:
    f.write(content_new)
