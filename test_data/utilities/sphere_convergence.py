import xml.etree.ElementTree as ET
import subprocess
import os
import re
import math
import sys


def check_bounds(porder, rate):
    if (porder=="1"):
        if rate<3 and rate>1:
            return True
        else:
            return False
    if (porder=="2"):
        if rate<4 and rate>1:
            return True
        else:
            return False
    if (porder=="3"):
        if rate<5 and rate>3:
            return True
        else:
            return False

# first argument sets maximum porder to check
# second argument sets maximum meshes to check
max_porder = 10
max_fname = 10 
use_obfet = 1
if (len(sys.argv) > 1):
    max_porder = int(sys.argv[1])
if (len(sys.argv) > 2):
    max_fname = int(sys.argv[2])
if (len(sys.argv) > 3):
    use_obfet = int(sys.argv[3])

porders = ["1","2","3","4","5"]
#porders = ["2",]
#porders = ["3",]
#file_names = ["mpas_2562.nc", "mpas_10242.nc"]
file_names = ["mpas_2562.nc", "mpas_10242.nc", "mpas_40962.nc", "mpas_163842.nc"]
#file_names = ["mpas_2562.nc", ]
sw_case2_errors = []
sphere_velocity_errors = [] 
for key1, porder in enumerate(porders):
    for key2, fname in enumerate(file_names):
        e = ET.parse('../test_data/parameter_lists/canga/parameters_template.xml').getroot()
        tree = ET.ElementTree(e)
        
        for item in e.getchildren():
            if (item.attrib['name']=="io"):
                f=item
            if (item.attrib['name']=="remap"):
                g=item
            if (item.attrib['name']=="halo"):
                h=item
        
        # have io now
        for item in f.getchildren():
            if (item.attrib['name']=="input file"):
                item.attrib['value']=fname;

        # have io now
        for item in g.getchildren():
            if (item.attrib['name']=="porder"):
                item.attrib['value']=porder;
            if (item.attrib['name']=="curvature porder"):
                item.attrib['value']=porder;
            if (item.attrib['name']=="obfet"):
                if (use_obfet):
                    item.attrib['value']="true";
                else:
                    item.attrib['value']="false";


        # have io now
        for item in h.getchildren():
            if (item.attrib['name']=="multiplier"):
                if (porder=="1"):
                    item.attrib['value']="20.0";
        
        tree.write(open('../test_data/parameter_lists/canga/parameters_lower_1.xml', 'wb'))
        
        # overwrite
        e = ET.parse('../test_data/parameter_lists/canga/parameters_lower_1.xml').getroot()
        for item in e.getchildren():
            if (item.attrib['name']=="my coloring"):
                item.attrib['value']="33";
            if (item.attrib['name']=="peer coloring"):
                item.attrib['value']="25";
        
        tree = ET.ElementTree(e)
        tree.write(open('../test_data/parameter_lists/canga/parameters_upper_1.xml', 'wb'))
        
        if (use_obfet):
            output = subprocess.check_output(["mpirun", "-np", "1", "./cangaRemoteRemap.exe","--i=../test_data/parameter_lists/canga/parameters_lower_1.xml","--kokkos-threads=1",":","-np","1","./cangaRemoteRemap.exe","--i=../test_data/parameter_lists/canga/parameters_upper_1.xml","--kokkos-threads=1"])
        else:
            output = subprocess.check_output(["mpirun", "-np", "5", "./cangaRemoteRemap.exe","--i=../test_data/parameter_lists/canga/parameters_lower_1.xml","--kokkos-threads=1",":","-np","3","./cangaRemoteRemap.exe","--i=../test_data/parameter_lists/canga/parameters_upper_1.xml","--kokkos-threads=1"])
        #output = subprocess.check_output(["mpirun", "-np", "2", "../bin/cangaRemoteRemap.exe","--i=canga/parameters_lower_1.xml","--kokkos-threads=1",":","-np","3","../bin/cangaRemoteRemap.exe","--i=canga/parameters_upper_1.xml","--kokkos-threads=1"])
        #print output
        m = re.search('(?<=Global Norm of Shallow Water Test Case 2: )[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', output)
        sw_case2_errors.append(float(m.group(0)))
        m = re.search('(?<=Global Norm of Spherical Velocity: )[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', output)
        sphere_velocity_errors.append(float(m.group(0)))
        
        if (max_fname==(key2+1)):
            break
    

    #print height_errors
    print "\n\nShallow Water Test Case 2 Rates: POrder:%s\n============="%porder
    for i in range(1,len(sw_case2_errors)):
        if (sw_case2_errors[i]!=0):
            rate = math.log(sw_case2_errors[i]/sw_case2_errors[i-1])/math.log(.5)
            print str(rate)
            #assert(check_bounds(porder, rate))
        else:
            print "NaN - Division by zero"
        
    #print velocity_errors
    print "\n\nSphere Velocity Rates: POrder:%s\n==============="%porder
    for i in range(1,len(sphere_velocity_errors)):
        if (sphere_velocity_errors[i]!=0):
            rate = math.log(sphere_velocity_errors[i]/sphere_velocity_errors[i-1])/math.log(.5)
            print str(rate)
            #assert(check_bounds(porder, rate))
        else:
            print "NaN - Division by zero"
  
    sw_case2_errors = []
    sphere_velocity_errors = []
    
    if (porder==str(max_porder)):
        break

sys.exit(0)
