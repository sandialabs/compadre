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
solution_type_num = 0
if (len(sys.argv) > 1):
    max_porder = int(sys.argv[1])
if (len(sys.argv) > 2):
    max_fname = int(sys.argv[2])
if (len(sys.argv) > 3):
    solution_type_num = int(sys.argv[3])
if (len(sys.argv) > 4):
    problem_setup = int(sys.argv[4])
else :
    problem_setup = 0

porders = ["%d"%num for num in range(2,11)]
solution_type_names = ["point","vector","grad","div_grad","staggered_div_grad","laplace","lb solve","staggered_grad","div","staggered_div","staggered_laplace"]
file_names=[]
if (problem_setup==0):
    file_names = ["shallow_%d.nc"%num for num in range(4)]
else:
    file_names = ["cylinder_%d.vtk"%num for num in range(5)]
errors = []
geometry=['sphere','cylinder','cylinder']
problem_num=['0','10','3']

for key1, porder in enumerate(porders):
    for key2, fname in enumerate(file_names):
        e = ET.parse('../test_data/parameter_lists/laplace_beltrami/parameters_template.xml').getroot()
        tree = ET.ElementTree(e)
        
        for item in e.getchildren():
            if (item.attrib['name']=="io"):
                f=item
            if (item.attrib['name']=="remap"):
                g=item
            if (item.attrib['name']=="solution type"):
                h=item
            if (item.attrib['name']=="physics number"):
                i=item
        
        # have io now
        for item in f.getchildren():
            if (item.attrib['name']=="input file"):
                item.attrib['value']=fname;

        # have remap now
        for item in g.getchildren():
            if (item.attrib['name']=="porder"):
                item.attrib['value']=porder;
            if (item.attrib['name']=="curvature porder"):
                item.attrib['value']=porder;

        # have solution type now
        h.attrib['value']=solution_type_names[solution_type_num];

        # have physics number now
        i.attrib['value']=problem_num[problem_setup]
        
        tree.write(open('../test_data/parameter_lists/laplace_beltrami/parameters_'+solution_type_names[solution_type_num]+'.xml', 'wb'))
        
        #subprocess.call(['C:\\Temp\\a b c\\Notepad.exe', 'C:\\test.txt'])
        num_procs = '2' if problem_setup == 0 else ('2' if int(porder)<4 else '1')
        with open(os.devnull, 'w') as devnull:
            output = subprocess.check_output(["mpirun", "-np", "2", "./laplaceBeltrami.exe","--i=../test_data/parameter_lists/laplace_beltrami/parameters_"+solution_type_names[solution_type_num]+".xml","--kokkos-threads=4"], stderr=devnull)
            #print output
            m = re.search('(?<=Global Norm: )[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', output)
            errors.append(float(m.group(0)))
        
        if (max_fname==(key2+1)):
            break
    
    #print errors
    print "\n\n%s on %s error rates: porder:%s\n============="%(solution_type_names[solution_type_num],geometry[problem_setup],porder)
    for i in range(1,len(errors)):
        if (errors[i]!=0):
            rate = math.log(errors[i]/errors[i-1])/math.log(.5)
            print str(rate) + ", " + str(errors[i]) + ", " + str(errors[i-1])
            assert(check_bounds(porder, rate))
        else:
            print "NaN - Division by zero"
        
    errors = []
    
    if (porder==str(max_porder)):
        break

sys.exit(0)
