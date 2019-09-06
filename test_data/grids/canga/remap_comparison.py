import xml.etree.ElementTree as ET
import subprocess
import os
import re
import math
import sys
from consolidate import *

assert len(sys.argv)>0, "Not enough arguments"

total_iterations = 10
opt_num = 1
mesh_1 = 0
mesh_2 = 0
start_step = 1

if (len(sys.argv) > 1):
    total_iterations = int(sys.argv[1])

if (len(sys.argv) > 2):
    opt_num = int(sys.argv[2])

if (len(sys.argv) > 3):
    mesh_1 = int(sys.argv[3])

if (len(sys.argv) > 4):
    mesh_2 = int(sys.argv[4])

if (len(sys.argv) > 5):
    start_step = int(sys.argv[5])

opt_types = ['NONE', 'OBFET', 'CAAS']
opt_name = opt_types[opt_num]

NPTS=[16,32,64,128,256]

pre_CS_name   = "../../test_data/grids/canga/Cubed-Sphere/outCSMesh_ne"
post_CS_name  = "_TPW_CFR_TPO.pvtp"
pre_CVT_name  = "../../test_data/grids/canga/CVT-MPAS/outICODMesh_ne"
post_CVT_name = ".pvtp"

# initially set to these existing files
CS_names = [pre_CS_name+str(i)+post_CS_name for i in NPTS]
CVT_names = [pre_CVT_name+str(i)+post_CVT_name for i in NPTS]

print(CS_names)
print(CVT_names)

def create_XML(file1, file2, file3):
    # goes from file1+file2 -> file3
    e = ET.parse('../test_data/parameter_lists/canga/parameters_comparison_template.xml').getroot()
    tree = ET.ElementTree(e)
    
    for item in e.getchildren():
        if (item.attrib['name']=="io"):
            f=item
        if (item.attrib['name']=="remap"):
            g=item
    
    # have io now
    for item in f.getchildren():
        if (item.attrib['name']=="input file prefix"):
            item.attrib['value']="./";
        if (item.attrib['name']=="input file"):
            item.attrib['value']=file1;
        if (item.attrib['name']=="target file prefix"):
            item.attrib['value']="./";
        if (item.attrib['name']=="target file"):
            item.attrib['value']=file2;
        if (item.attrib['name']=="output file prefix"):
            item.attrib['value']="./";
        if (item.attrib['name']=="output file"):
            item.attrib['value']=file3;
        if (item.attrib['name']=="alt output file prefix"):
            item.attrib['value']="./";
        if (item.attrib['name']=="alt output file"):
            item.attrib['value']=file3+".g";

    # have io now
    for item in g.getchildren():
        if (item.attrib['name']=="optimization algorithm"):
            item.attrib['value']=opt_name;
    
    tree.write(open('../test_data/parameter_lists/canga/parameters_current.xml', 'wb'))

def run_transfer(opt_name):
    try:
        if (opt_name=="OBFET"): # OBFET only works in serial
            my_env=dict(os.environ)
            output = subprocess.check_output(["mpirun", "-np", "1", "./realCangaIntercomparison.exe","--i=../test_data/parameter_lists/canga/parameters_current.xml","--kokkos-threads=32"], env=my_env).decode()
        else:
            my_env=dict(os.environ)
            my_env['OMP_PROC_BIND']='spread'
            my_env['OMP_PLACES']='threads'
            my_env['OMP_NUM_THREADS']='1'
            output = subprocess.check_output(["./realCangaIntercomparison.exe","--i=../test_data/parameter_lists/canga/parameters_current.xml","--kokkos-threads=32"], env=my_env).decode()
            #output = subprocess.check_output(["mpirun", "-np", "1", "./realCangaIntercomparison.exe","--i=../test_data/parameter_lists/canga/parameters_current.xml","--kokkos-threads=32"], env=os.environ{"OMP_PROC_BIND": "spread", "OMP_PLACES": "threads"}, shell=True).decode()
    except subprocess.CalledProcessError as exc:
        print("error code", exc.returncode)
        for line in exc.output.decode().split('\n'):
            print(line)
        sys.exit(exc.returncode)

f1 = CS_names[mesh_1]
f2 = CVT_names[mesh_2]

# get initial name from existing files
current_f1_name = f1 # copies by value
current_f2_name = f2

if (start_step > 1): # starting at 1 is the same as starting fresh
    current_f1_name = "backward_" + str(start_step-1) + ".pvtp"
    current_f2_name = "forward_" + str(start_step-1) + ".pvtp"

for step in range(start_step-1,total_iterations):
    step_num = step+1

    output_f = "forward_" + str(step_num) + ".pvtp"
    # do transfer from current_f1_name + current_f2_name and save as output_f
    # prepare XML file
    create_XML(current_f1_name, current_f2_name, output_f)
    print(current_f1_name + "+" + current_f2_name + "=" + output_f)

    # execute
    run_transfer(opt_name)

    # set current_f2_name=output_f (forward ...)
    current_f2_name = output_f

    output_f = "backward_" + str(step_num) + ".pvtp"
    # do transfer from current_f2_name + current_f1_name and save as output_f
    # prepare XML file
    create_XML(current_f2_name, current_f1_name, output_f)
    print(current_f2_name + "+" + current_f1_name + "=" + output_f)

    # execute
    run_transfer(opt_name)

    # set current_f1_name=output_f (backward ...)
    current_f1_name = output_f

# strip off pvtp and add .g to names
f1_head, f1_tail = os.path.splitext(f1)
f2_head, f2_tail = os.path.splitext(f2)
print(f1_head,f2_head)
consolidate(total_iterations, f1_head+'.g', f2_head+'.g')

