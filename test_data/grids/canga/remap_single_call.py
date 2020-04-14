import xml.etree.ElementTree as ET
import subprocess
import os
import re
import math
import sys
from consolidate import *
from reorder_file_by_field import *
import time

assert len(sys.argv)>0, "Not enough arguments"

'''

Copies solutions from pre_CS_name grids to pre_CVT_name grids and then back
total_iterations number of times

The executable is only called ONCE. The neighbor searches and GMLS solution is 
performed once and saved and reused each following iteration.

'''

total_iterations = 10
opt_num = 1
mesh_1 = 0
mesh_2 = 0
start_step = 1
porder = 4

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

if (len(sys.argv) > 6):
    porder = int(sys.argv[6])

opt_types = ['NONE', 'OBFET', 'CAAS']
opt_name = opt_types[opt_num]

NPTS=[16,32,64,128,256]

pre_CS_name   = "../../test_data/grids/canga/NM16/CS_"
post_CS_name  = ".g"
pre_CVT_name  = "../../test_data/grids/canga/NM16/ICOD_"
post_CVT_name = ".g"

# initially set to these existing files
CS_names = [pre_CS_name+str(i)+post_CS_name for i in NPTS]
CVT_names = [pre_CVT_name+str(i)+post_CVT_name for i in NPTS]

#print(CS_names)
#print(CVT_names)

def create_XML(file1, file2, total_steps):
    e = ET.parse('../../test_data/parameter_lists/canga/parameters_comparison_template.xml').getroot()

    tree = ET.ElementTree(e)
    for item in e.getchildren():
        if (item.attrib['name']=="my coloring"):
            item.attrib['value']="25";
        if (item.attrib['name']=="peer coloring"):
            item.attrib['value']="33";
        if (item.attrib['name']=="initial step"):
            item.attrib['value']="1";
        if (item.attrib['name']=="final step"):
            item.attrib['value']=str(total_steps);
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
        if (item.attrib['name']=="output file prefix"):
            item.attrib['value']="./";
        if (item.attrib['name']=="output file"):
            item.attrib['value']=file2;

    # have remap now
    for item in g.getchildren():
        if (item.attrib['name']=="optimization algorithm"):
            item.attrib['value']=opt_name;
        if (item.attrib['name']=="porder"):
            item.attrib['value']=str(porder);
        if (item.attrib['name']=="curvature porder"):
            item.attrib['value']=str(porder);
    tree.write(open('../test_data/parameter_lists/canga/parameters_comparison_25.xml', 'wb'))

    tree = ET.ElementTree(e)
    for item in e.getchildren():
        if (item.attrib['name']=="my coloring"):
            item.attrib['value']="33";
        if (item.attrib['name']=="peer coloring"):
            item.attrib['value']="25";
        if (item.attrib['name']=="initial step"):
            item.attrib['value']="1";
        if (item.attrib['name']=="final step"):
            item.attrib['value']=str(total_steps);
        if (item.attrib['name']=="io"):
            f=item
        if (item.attrib['name']=="remap"):
            g=item
    
    # have io now
    for item in f.getchildren():
        if (item.attrib['name']=="input file prefix"):
            item.attrib['value']="./";
        if (item.attrib['name']=="input file"):
            item.attrib['value']=file2;
        if (item.attrib['name']=="output file prefix"):
            item.attrib['value']="./";
        if (item.attrib['name']=="output file"):
            item.attrib['value']=file1;

    # have io now
    for item in g.getchildren():
        if (item.attrib['name']=="optimization algorithm"):
            item.attrib['value']=opt_name;
        if (item.attrib['name']=="porder"):
            item.attrib['value']=str(porder);
        if (item.attrib['name']=="curvature porder"):
            item.attrib['value']=str(porder);
    tree.write(open('../test_data/parameter_lists/canga/parameters_comparison_33.xml', 'wb'))

def run_transfer(opt_name):
    try:
        if (opt_name=="OBFET"): # OBFET only works in serial
            my_env=dict(os.environ)
            output = subprocess.check_output(["mpirun", "-np", "1", "./cangaRemoteRemapMultiIter.exe","--i=../test_data/parameter_lists/canga/parameters_comparison_25.xml",":","-np", "1", "./cangaRemoteRemapMultiIter.exe","--i=../test_data/parameter_lists/canga/parameters_comparison_33.xml","--kokkos-threads=32"], env=my_env)
        else:
            my_env=dict(os.environ)
            #my_env['OMP_PROC_BIND']='spread'
            #my_env['OMP_PLACES']='threads'
            #my_env['OMP_NUM_THREADS']='1'
            #output = subprocess.check_output(["mpirun", "--bind-to", "socket", "-np", "4", "./cangaRemoteRemapMultiIter.exe","--i=../test_data/parameter_lists/canga/parameters_comparison_25.xml","--kokkos-threads=4",":","-np", "4", "./cangaRemoteRemapMultiIter.exe","--i=../test_data/parameter_lists/canga/parameters_comparison_33.xml","--kokkos-threads=4"], env=my_env).decode()
            output = subprocess.check_output(["mpirun", "--bind-to", "none", "-np", "1", "./cangaRemoteRemapMultiIter.exe","--i=../test_data/parameter_lists/canga/parameters_comparison_25.xml","--kokkos-threads=7",":","-np", "1", "./cangaRemoteRemapMultiIter.exe","--i=../test_data/parameter_lists/canga/parameters_comparison_33.xml","--kokkos-threads=7"], env=my_env)
            output_str = str(output)
            print(output_str.replace("\\n","\n"))
    except subprocess.CalledProcessError as exc:
        print("error code", exc.returncode)
        for line in exc.output.decode().split('\n'):
            print(line)
        sys.exit(exc.returncode)

f1 = CS_names[mesh_1]
f2 = CVT_names[mesh_2]

print(f1,f2)
# get initial name from existing files
current_f1_name = f1 # copies by value
current_f2_name = f2
create_XML(current_f1_name, current_f2_name, total_iterations)

# execute
run_transfer(opt_name)

# strip off pvtp and add .g to names
f1_head, f1_tail = os.path.splitext(f1)
f2_head, f2_tail = os.path.splitext(f2)
#print(f1_head,f2_head)
time.sleep(1)
newname=consolidate(total_iterations, f1_head+'.g', f2_head+'.g')
reorder_file_by_field(newname,'_remap_src')
reorder_file_by_field(newname,'_remap_tgt')

