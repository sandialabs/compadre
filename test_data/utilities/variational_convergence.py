import xml.etree.ElementTree as ET
import subprocess
import os
import re
import math
import sys


# first argument sets maximum porder to check
# second argument sets maximum meshes to check
max_fname = 10 
if (len(sys.argv) > 1):
    max_fname = int(sys.argv[1])

orig_size = 1.0
if (len(sys.argv) > 2):
    orig_size = float(sys.argv[2])

file_names = ["dg_%d.nc"%num for num in range(max_fname)]
error_types=['l2','h1','jp','sum']
all_errors = [list(), list(), list(), list()]#list() * len(error_types)

for key2, fname in enumerate(file_names):
    e = ET.parse('../test_data/parameter_lists/reactiondiffusion/parameters.xml').getroot()
    tree = ET.ElementTree(e)
    size_str = str(orig_size/float(pow(2,key2)))
    print(size_str)
    
    for item in e.getchildren():
        if (item.attrib['name']=="io"):
            f=item
        if (item.attrib['name']=="neighborhood"):
            n=item
    #    if (item.attrib['name']=="remap"):
    #        g=item
    #    if (item.attrib['name']=="physics"):
    #        p=item
    
    # have io now
    for item in f.getchildren():
        if (item.attrib['name']=="input file"):
            item.attrib['value']=fname

    for item in n.getchildren():
        if (item.attrib['name']=="size"):
            item.attrib['value']=size_str

    ## have physics now
    #for item in p.getchildren():
    #    if (item.attrib['name']=="operator"):
    #        item.attrib['value']="l2" if (l2_only > 0) else "rd"
    #    if (item.attrib['name']=="solution"):
    #        item.attrib['value']="polynomial" if (solution==0) else "sine"
    
    tree.write(open('../test_data/parameter_lists/reactiondiffusion/parameters_generated.xml', 'wb'))
    
    with open(os.devnull, 'w') as devnull:

        commands = ["./reactionDiffusion.exe","--i=../test_data/parameter_lists/reactiondiffusion/parameters_generated.xml","--kokkos-threads=8"]
        print(" ".join(commands))
        try:
            output = subprocess.check_output(commands, stderr=devnull).decode()
        except subprocess.CalledProcessError as exc:
            print("error code", exc.returncode)
            for line in exc.output.decode().split('\n'):
                print(line)
            sys.exit(exc.returncode)
        print(output)
        m = re.search('(?<=L2: )[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', output)
        all_errors[0].append(float(m.group(0)))
        m = re.search('(?<=H1: )[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', output)
        all_errors[1].append(float(m.group(0)))
        m = re.search('(?<=Ju: )[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', output)
        all_errors[2].append(float(m.group(0)))
        m = re.search('(?<=Global Norm: )[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', output)
        all_errors[3].append(float(m.group(0)))
    
    if (max_fname==(key2+1)):
        break

print(all_errors)
for key, errors in enumerate(all_errors):
    if (max_fname>1):
        print("\n\nerror rates: type:%s\n============="%(error_types[key],))
        for i in range(1,len(errors)):
            if (errors[i]!=0):
                rate = math.log(errors[i]/errors[i-1])/math.log(.5)
                print(str(rate) + ", " + str(errors[i]) + ", " + str(errors[i-1]))
            else:
                print("NaN - Division by zero")

sys.exit(0)
