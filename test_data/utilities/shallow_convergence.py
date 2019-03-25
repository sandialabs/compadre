import xml.etree.ElementTree as ET
import subprocess
import os
import re
import math

os.chdir("./tests")


time_step = ["100","50","25"]; t_end=100;
#time_step = ["3600","1800",]#"900"]; t_end=10000;
file_names = ["shallow_0.nc", "shallow_1.nc",]# "shallow_2.nc"]
height_errors = []
velocity_errors = []

for (timestep, fname)  in zip(time_step, file_names):
    e = ET.parse('./shallow_water/parameters_basic.xml').getroot()
    tree = ET.ElementTree(e)
    
    for item in e.getchildren():
        if (item.attrib['name']=="time"):
            g=item
        if (item.attrib['name']=="io"):
            f=item
    
    # have io now
    for item in f.getchildren():
        if (item.attrib['name']=="input file"):
            item.attrib['value']=fname;

    # have io now
    for item in g.getchildren():
        if (item.attrib['name']=="dt"):
            item.attrib['value']=timestep;
        if (item.attrib['name']=="t_end"):
            item.attrib['value']=str(t_end);
    
    tree.write(open('./parameters_generated.xml', 'wb'))
    
    #subprocess.call(["mpirun", "-np", "1", "../bin/lagrangianShallowWater.exe","--i=parameters_generated.xml","--kokkos-threads=1", "|", "tee", "out.txt"])
    output = subprocess.check_output(["mpirun", "-np", "1", "../bin/lagrangianShallowWater.exe","--i=parameters_generated.xml","--kokkos-threads=1"], encoding='UTF-8')
    print(output)
    m = re.search('(?<=Global relative VELOCITY error: )[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', output)
    velocity_errors.append(float(m.group(0)))
    m = re.search('(?<=Global relative HEIGHT error: )[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', output)
    height_errors.append(float(m.group(0)))

#print height_errors
print("HEIGHT Rates:\n=============")
for i in range(1,len(height_errors)):
    if (height_errors[i]!=0):
        print(str(math.log(height_errors[i]/height_errors[i-1])/math.log(.5)))
    else:
        print("NaN - Division by zero")
    
#print velocity_errors
print("\n\nVELOCITY Rates:\n===============")
for i in range(1,len(velocity_errors)):
    if (velocity_errors[i]!=0):
        print(str(math.log(velocity_errors[i]/velocity_errors[i-1])/math.log(.5)))
    else:
        print("NaN - Division by zero")
