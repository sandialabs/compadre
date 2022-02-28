import xml.etree.ElementTree as ET
import subprocess
import os
import re
import math
import sys
from consolidate import *
from reorder_file_by_field import *
import time


'''

Copies solutions from pre_CS_name grids to pre_CVT_name grids and then back
total_iterations number of times

The executable is only called ONCE. The neighbor searches and GMLS solution is 
performed once and saved and reused each following iteration.

'''

import argparse

parser = argparse.ArgumentParser(description='Run CANGA example for remap comparison.')

parser.add_argument('--total-iterations', dest='total_iterations', type=int, default=10, nargs='?', help='total round trip remaps (forth and back)')
parser.add_argument('--save-every', dest='save_every', type=int, default=1, nargs='?', help='store every n iterations')
parser.add_argument('--mesh-1', dest='mesh_1', type=int, default=0, nargs='?', help='mesh 1 number')
parser.add_argument('--mesh-1-type', dest='mesh_1_type', type=str, default="CS", nargs='?', help='mesh 1 type {CS,CVT,RLL}')
parser.add_argument('--mesh-2', dest='mesh_2', type=int, default=0, nargs='?', help='mesh 2 number')
parser.add_argument('--mesh-2-type', dest='mesh_2_type', type=str, default="CVT", nargs='?', help='mesh 1 type {CS,CVT,RLL}')
parser.add_argument('--start-step', dest='start_step', type=int, default=1, nargs='?', help='starting step number (for continuing off of checkpoints)')
parser.add_argument('--optimization', dest='optimization', type=str, nargs='?', default='NONE', help='optimization algorithm {NONE,OBFET,CAAS}')
parser.add_argument('--porder', dest='porder', type=int, nargs='?', default='4', help='polynomial order')
parser.add_argument('--exe-folder-relative', dest='exe_folder_relative', type=str, nargs='?', default='', help='where to get executable')
parser.add_argument('--exe-folder-absolute', dest='exe_folder_absolute', type=str, nargs='?', default='', help='where to get executable')
parser.add_argument('--output-folder-relative', dest='output_folder_relative', type=str, nargs='?', default='', help='where to dump data files (relative to directory this script is called from')
parser.add_argument('--output-folder-absolute', dest='output_folder_absolute', type=str, nargs='?', default='', help='where to dump data files (relative to directory this script is called from')
parser.add_argument('--canga-folder-relative', dest='canga_folder_relative', type=str, nargs='?', default="", help='where to get canga meshes')
parser.add_argument('--canga-folder-absolute', dest='canga_folder_absolute', type=str, nargs='?', default="", help='where to get canga meshes')
parser.add_argument('--preserve-local-bounds', dest='preserve_local_bounds', type=str, nargs='?', default='false', help='whether to preserve local bounds in optimization')
parser.add_argument('--batches', dest='batches', type=int, default=1, nargs='?', help='number of batches to break target sites into')

args = parser.parse_args()

if args.canga_folder_absolute!="":
    canga_folder = args.canga_folder_absolute
else:
    canga_folder = os.getcwd() + "/" + args.canga_folder_relative
print("canga_folder:",canga_folder)

if args.output_folder_absolute!="":
    output_folder = args.output_folder_absolute
else:
    output_folder = os.getcwd() + "/" + args.output_folder_relative
print("output_folder:",output_folder)

if args.exe_folder_absolute!="":
    exe_folder = args.exe_folder_absolute
else:
    exe_folder = os.getcwd() + "/" + args.exe_folder_relative
print("exe_folder:",exe_folder)

# store files with data
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

opt_types = ['NONE', 'OBFET', 'CAAS']

total_iterations = args.total_iterations
opt_num = opt_types.index(args.optimization)
opt_name = opt_types[opt_num]
mesh_1 = args.mesh_1
mesh_2 = args.mesh_2
start_step = args.start_step
porder = args.porder

#NPTS1=[32,64,128,256]
#NPTS2=[16,32,64,128]
CS_CVT_NPTS=[16,32,64,128,256]
RRM_CS_CVT_NPTS=[32,64,128]
RLL_NPTS=['30-60','90-180','180-360','360-720','720-1440']

CS_CVT_LVL=1
RLL_LVL=1
RRM_CS_LVL=1
RRM_CVT_LVL=0

pre_CS_name   = "CS_"
post_CS_name  = ".g"
pre_CVT_name  = "ICOD_"
post_CVT_name = ".g"
pre_RRM_CS_name   = "RRM_CS_"
post_RRM_CS_name  = ".g"
pre_RRM_CVT_name  = "RRM_ICOD_"
post_RRM_CVT_name = ".g"
pre_RLL_name  = "RLL_"
post_RLL_name = ".g"
#pre_CS_name   = "../../test_data/grids/canga/NM16/ICOD_"
#post_CS_name  = ".g"
#pre_CVT_name  = "../../test_data/grids/canga/NM16/ICOD_"
#post_CVT_name = ".g"

# initially set to these existing files
#CS_names = [pre_CS_name+str(i)+post_CS_name for i in NPTS1]
#CVT_names = [pre_CVT_name+str(i)+post_CVT_name for i in NPTS2]
CS_names = [pre_CS_name+str(i)+post_CS_name for i in CS_CVT_NPTS]
CVT_names = [pre_CVT_name+str(i)+post_CVT_name for i in CS_CVT_NPTS]
RLL_names = [pre_RLL_name+i+post_RLL_name for i in RLL_NPTS]
RRM_CS_names = [pre_RRM_CS_name+str(i)+post_RRM_CS_name for i in RRM_CS_CVT_NPTS]
RRM_CVT_names = [pre_RRM_CVT_name+str(i)+post_RRM_CVT_name for i in RRM_CS_CVT_NPTS]

#print(CS_names)
#print(CVT_names)

def create_XML(file1, file2, total_steps):
    e = ET.parse(canga_folder+'/../../../parameter_lists/canga/parameters_comparison_template.xml').getroot()

    tree = ET.ElementTree(e)
    for item in list(e):
        if (item.attrib['name']=="initial step"):
            item.attrib['value']=str(start_step);
        if (item.attrib['name']=="save every"):
            item.attrib['value']=str(args.save_every);
        if (item.attrib['name']=="final step"):
            item.attrib['value']=str(total_steps);
        if (item.attrib['name']=="io"):
            f=item
        if (item.attrib['name']=="remap"):
            g=item
    
    # have io now
    for item in list(f):
        if (item.attrib['name']=="input file prefix"):
            item.attrib['value']=canga_folder+"/"
        if (item.attrib['name']=="input file 1"):
            item.attrib['value']=file1;
        if (item.attrib['name']=="input file 2"):
            item.attrib['value']=file2;
        if (item.attrib['name']=="output file prefix"):
            item.attrib['value']=output_folder+"/"
        if (item.attrib['name']=="output file"):
            item.attrib['value']="";

    # have remap now
    for item in list(g):
        if (item.attrib['name']=="optimization algorithm"):
            item.attrib['value']=opt_name;
        if (item.attrib['name']=="preserve local bounds"):
            item.attrib['value']=args.preserve_local_bounds.lower();
        if (item.attrib['name']=="porder"):
            item.attrib['value']=str(porder);
        if (item.attrib['name']=="curvature porder"):
            item.attrib['value']=str(porder);
        if (item.attrib['name']=="number of batches"):
            item.attrib['value']=str(args.batches);
    tree.write(open(output_folder+'/parameters_comparison.xml', 'wb'))

#def create_XML(file1, file2, total_steps):
#    e = ET.parse(canga_folder+'/../../../parameter_lists/canga/parameters_comparison_template.xml').getroot()
#
#    tree = ET.ElementTree(e)
#    for item in list(e):
#        if (item.attrib['name']=="my coloring"):
#            item.attrib['value']="25";
#        if (item.attrib['name']=="peer coloring"):
#            item.attrib['value']="33";
#        if (item.attrib['name']=="initial step"):
#            item.attrib['value']=str(start_step);
#        if (item.attrib['name']=="save every"):
#            item.attrib['value']=str(args.save_every);
#        if (item.attrib['name']=="final step"):
#            item.attrib['value']=str(total_steps);
#        if (item.attrib['name']=="io"):
#            f=item
#        if (item.attrib['name']=="remap"):
#            g=item
#    
#    # have io now
#    for item in list(f):
#        if (item.attrib['name']=="input file prefix"):
#            item.attrib['value']=canga_folder+"/"
#        if (item.attrib['name']=="input file"):
#            item.attrib['value']=file1;
#        if (item.attrib['name']=="output file prefix"):
#            item.attrib['value']=output_folder+"/"
#        if (item.attrib['name']=="output file"):
#            item.attrib['value']=file2;
#
#    # have remap now
#    for item in list(g):
#        if (item.attrib['name']=="optimization algorithm"):
#            item.attrib['value']=opt_name;
#        if (item.attrib['name']=="porder"):
#            item.attrib['value']=str(porder);
#        if (item.attrib['name']=="curvature porder"):
#            item.attrib['value']=str(porder);
#    tree.write(open(output_folder+'/parameters_comparison_25.xml', 'wb'))
#
#    tree = ET.ElementTree(e)
#    for item in list(e):
#        if (item.attrib['name']=="my coloring"):
#            item.attrib['value']="33";
#        if (item.attrib['name']=="peer coloring"):
#            item.attrib['value']="25";
#        if (item.attrib['name']=="initial step"):
#            item.attrib['value']=str(start_step);
#        if (item.attrib['name']=="final step"):
#            item.attrib['value']=str(total_steps);
#        if (item.attrib['name']=="io"):
#            f=item
#        if (item.attrib['name']=="remap"):
#            g=item
#    
#    # have io now
#    for item in list(f):
#        if (item.attrib['name']=="input file prefix"):
#            item.attrib['value']=canga_folder+"/"
#        if (item.attrib['name']=="input file"):
#            item.attrib['value']=file2;
#        if (item.attrib['name']=="output file prefix"):
#            item.attrib['value']=output_folder+"/"
#        if (item.attrib['name']=="output file"):
#            item.attrib['value']=file1;
#
#    # have remap now
#    for item in list(g):
#        if (item.attrib['name']=="optimization algorithm"):
#            item.attrib['value']=opt_name;
#        if (item.attrib['name']=="porder"):
#            item.attrib['value']=str(porder);
#        if (item.attrib['name']=="curvature porder"):
#            item.attrib['value']=str(porder);
#    tree.write(open(output_folder+'/parameters_comparison_33.xml', 'wb'))

def run_transfer(opt_name):
    try:
        if (opt_name=="OBFET"): # OBFET only works in serial
            my_env=dict(os.environ)
            #commands = ["mpirun", "-np", "1", exe_folder+"/cangaRemoteRemapMultiIter.exe","--i="+output_folder+"/parameters_comparison_25.xml",":","-np", "1", exe_folder+"/cangaRemoteRemapMultiIter.exe","--i="+output_folder+"/parameters_comparison_33.xml","--kokkos-threads=32"]
            commands = [exe_folder+"/cangaRemapMultiIter.exe","--i="+output_folder+"/parameters_comparison.xml","--kokkos-threads=32"]
            print(" ".join(commands))
            output = subprocess.check_output(commands, env=my_env)
        else:
            my_env=dict(os.environ)
            #commands = ["mpirun", "--bind-to", "none", "-np", "1", exe_folder+"/cangaRemoteRemapMultiIter.exe","--i="+output_folder+"/parameters_comparison_25.xml","--kokkos-threads=7",":","-np", "1", exe_folder+"/cangaRemoteRemapMultiIter.exe","--i="+output_folder+"/parameters_comparison_33.xml","--kokkos-threads=7"]
            commands = [exe_folder+"/cangaRemapMultiIter.exe","--i="+output_folder+"/parameters_comparison.xml","--kokkos-threads=32"]
            print(" ".join(commands))
            #my_env['OMP_PROC_BIND']='spread'
            #my_env['OMP_PLACES']='threads'
            #my_env['OMP_NUM_THREADS']='1'
            #output = subprocess.check_output(["mpirun", "--bind-to", "socket", "-np", "4", "./cangaRemoteRemapMultiIter.exe","--i=../test_data/parameter_lists/canga/parameters_comparison_25.xml","--kokkos-threads=4",":","-np", "4", "./cangaRemoteRemapMultiIter.exe","--i=../test_data/parameter_lists/canga/parameters_comparison_33.xml","--kokkos-threads=4"], env=my_env).decode()
            output = subprocess.check_output(commands, env=my_env)
            output_str = str(output)
            print(output_str.replace("\\n","\n"))
    except subprocess.CalledProcessError as exc:
        print("error code", exc.returncode)
        for line in exc.output.decode().split('\n'):
            print(line)
        sys.exit(exc.returncode)

blk_level1=1
if (args.mesh_1_type.lower()=="cs"):
    f1 = CS_names[mesh_1]
    blk_level1=CS_CVT_LVL
elif (args.mesh_1_type.lower()=="cvt"):
    f1 = CVT_names[mesh_1]
    blk_level1=CS_CVT_LVL
elif (args.mesh_1_type.lower()=="rll"):
    f1 = RLL_names[mesh_1]
    blk_level1=RLL_LVL
elif (args.mesh_1_type.lower()=="rrmcs"):
    f1 = RRM_CS_names[mesh_1]
    blk_level1=RRM_CS_LVL
elif (args.mesh_1_type.lower()=="rrmcvt"):
    f1 = RRM_CVT_names[mesh_1]
    blk_level1=RRM_CVT_LVL

blk_level2=1
if (args.mesh_2_type.lower()=="cs"):
    f2 = CS_names[mesh_2]
    blk_level2=CS_CVT_LVL
elif (args.mesh_2_type.lower()=="cvt"):
    f2 = CVT_names[mesh_2]
    blk_level2=CS_CVT_LVL
elif (args.mesh_2_type.lower()=="rll"):
    f2 = RLL_names[mesh_2]
    blk_level2=RLL_LVL
elif (args.mesh_2_type.lower()=="rrmcs"):
    f2 = RRM_CS_names[mesh_2]
    blk_level2=RRM_CS_LVL
elif (args.mesh_2_type.lower()=="rrmcvt"):
    f2 = RRM_CVT_names[mesh_2]
    blk_level2=RRM_CVT_LVL

print("files:",f1,f2)
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
newname=consolidate(total_iterations, args.save_every, canga_folder+"/"+f1_head+'.g', canga_folder+"/"+f2_head+'.g', output_folder, blk_level1, blk_level2)
#reorder_file_by_field(newname,'_remap_src')
#reorder_file_by_field(newname,'_remap_tgt')

