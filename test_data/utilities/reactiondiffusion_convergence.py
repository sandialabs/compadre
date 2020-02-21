import xml.etree.ElementTree as ET
import subprocess
import os
import re
import math
import sys


def check_bounds(porder, rate):
    if (porder=="1"):
        if rate<3 and rate>.9:
            return True
        else:
            return False
    else:
        if rate>float(porder)-1.1:
            return True
        else:
            return False

# first argument sets maximum porder to check
# second argument sets maximum meshes to check
max_porder = 10
max_fname = 10 
l2_only = 0
solution = 0
if (len(sys.argv) > 1):
    max_porder = int(sys.argv[1])
if (len(sys.argv) > 2):
    max_fname = int(sys.argv[2])
if (len(sys.argv) > 3):
    l2_only = int(sys.argv[3])
if (len(sys.argv) > 4):
    solution = int(sys.argv[4])

porders = ["%d"%num for num in range(1,11)]
file_names = ["dg_%d.nc"%num for num in range(4)]
errors = []

for key1, porder in enumerate(porders):
    for key2, fname in enumerate(file_names):
        e = ET.parse('../test_data/parameter_lists/reactiondiffusion/parameters_template.xml').getroot()
        tree = ET.ElementTree(e)
        
        for item in e.getchildren():
            if (item.attrib['name']=="io"):
                f=item
            if (item.attrib['name']=="remap"):
                g=item
            if (item.attrib['name']=="physics"):
                p=item
        
        # have io now
        for item in f.getchildren():
            if (item.attrib['name']=="input file"):
                item.attrib['value']=fname

        # have remap now
        for item in g.getchildren():
            if (item.attrib['name']=="porder"):
                item.attrib['value']=porder

        # have physics now
        for item in p.getchildren():
            if (item.attrib['name']=="operator"):
                item.attrib['value']="l2" if (l2_only > 0) else "rd"
            if (item.attrib['name']=="solution"):
                item.attrib['value']="polynomial" if (solution==0) else "sine"
        
        tree.write(open('../test_data/parameter_lists/reactiondiffusion/parameters.xml', 'wb'))
        
        with open(os.devnull, 'w') as devnull:

            commands = ["mpirun", "-np", "1", "./reactionDiffusion.exe","--i=../test_data/parameter_lists/reactiondiffusion/parameters.xml","--kokkos-threads=4"]
            print(" ".join(commands))
            try:
                output = subprocess.check_output(commands, stderr=devnull).decode()
            except subprocess.CalledProcessError as exc:
                print("error code", exc.returncode)
                for line in exc.output.decode().split('\n'):
                    print(line)
                sys.exit(exc.returncode)
            print(output)
            m = re.search('(?<=Global Norm: )[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', output)
            errors.append(float(m.group(0)))
        
        if (max_fname==(key2+1)):
            break
    
    if (max_fname>1 or solution!=0):
        #print errors
        print("\n\nerror rates: porder:%s\n============="%(porder,))
        for i in range(1,len(errors)):
            if (errors[i]!=0):
                rate = math.log(errors[i]/errors[i-1])/math.log(.5)
                print(str(rate) + ", " + str(errors[i]) + ", " + str(errors[i-1]))
                #assert(check_bounds(porder, rate))
            else:
                print("NaN - Division by zero")
    else: # 1 mesh and polynomial solution, so should be exact
        print("\n\nerror: porder:%s\n============="%(porder,))
        print(str(errors[0]))
        assert(errors[0]<1e-14, "Solution not exact to machine precision")
        
    errors = []
    
    if (porder==str(max_porder)):
        break

sys.exit(0)
