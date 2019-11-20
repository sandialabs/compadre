import sys
import re
import numpy as np

# parses file name given as command line argument, calculating convergence
# rates for various fields

assert len(sys.argv)>2, "Not enough arguments"
if (len(sys.argv) > 1):
    original_filename = str(sys.argv[1])

fields_requested = list()
arg_count = len(sys.argv)-2
for i in range(arg_count):
    fields_requested.append(str(sys.argv[2+i]))

f = open(original_filename, "r")
lines = f.readlines()
f.close()
#lines = ['Target Absolute l2 error cloudfraction 0.0367332409045112',
#'Target Absolute l2 error totalprecipwater 1.1746364382243577',
#'Target Absolute l2 error topography 490.1387753640281062']

def getRates(lines, line_prefix, field_name): 
    total_errors = 0
    errors = []
    for line in lines:
        my_regex = re.compile(r'^%s[^\d]*l2[^\d]*%s[^\d]*(\d+\.\d+).*'%(line_prefix,field_name))
        out = my_regex.findall(line)
        if (len(out) > 0):
            errors.append(out[0])
            total_errors += 1

    np_errors = np.zeros(shape=(total_errors,), dtype='f8')
    for i in range(np_errors.shape[0]):
        np_errors[i] = np.float64(errors[i])

    header_string = line_prefix+" "+field_name+":"
    print(header_string)
    print("="*len(header_string))
    for i in range(1,np_errors.shape[0]):
        print(np.log(np_errors[i]/np_errors[i-1])/np.log(0.5))

print("")
for field in fields_requested:
    getRates(lines, "Source", field)
    getRates(lines, "Target", field)
    print("")

