import xml.etree.ElementTree as ET
import subprocess
import os
import re
import math
import sys

assert len(sys.argv)>0, "Not enough arguments"

# call from inside of build/examples

NPTS=[16,32,64,128,256]

pre_CS_name   = "../../test_data/grids/canga/Cubed-Sphere/outCSMesh_ne"
post1_CS_name  = "_TPW_CFR_TPO"
post2_CS_name  = ".g"
pre_CVT_name  = "../../test_data/grids/canga/CVT-MPAS/outICODMesh_ne"
post1_CVT_name = ""
post2_CVT_name = ".g"

# initially set to these existing files
#CS_names = [pre_CS_name+str(i)+post_CS_name for i in NPTS]
#CVT_names = [pre_CVT_name+str(i)+post_CVT_name for i in NPTS]

def create_XML(pre,post,ext,num):
    e = ET.parse('../test_data/parameter_lists/canga/convert_nc_to_vtk_template.xml').getroot()
    tree = ET.ElementTree(e)
    
    for item in e.getchildren():
        if (item.attrib['name']=="io"):
            f=item
    
    # have io now
    for item in f.getchildren():
        if (item.attrib['name']=="input file prefix"):
            item.attrib['value']="./";
        if (item.attrib['name']=="input file"):
            item.attrib['value']=pre+num+post+ext;
        if (item.attrib['name']=="output file prefix"):
            item.attrib['value']="./";
        if (item.attrib['name']=="output file"):
            item.attrib['value']=pre+num+post+".pvtp";
    
    tree.write(open('../test_data/parameter_lists/canga/parameters_current.xml', 'wb'))

def run_transfer():
    try:
        output = subprocess.check_output(["mpirun", "-np", "1", "./cangaFileConverter.exe","--i=../test_data/parameter_lists/canga/parameters_current.xml","--kokkos-threads=1"]).decode()
    except subprocess.CalledProcessError as exc:
        print("error code", exc.returncode)
        for line in exc.output.decode().split('\n'):
            print(line)
        sys.exit(exc.returncode)

for i in NPTS:
    create_XML(pre_CS_name, post1_CS_name, post2_CS_name, str(i))
    run_transfer()

    create_XML(pre_CVT_name, post1_CVT_name, post2_CVT_name, str(i))
    run_transfer()

