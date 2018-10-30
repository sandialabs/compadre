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
    if (porder=="2"):
        if rate<4 and rate>.9:
            return True 
        else:
            return False
    if (porder=="3"):
        if rate<5 and rate>1.9:
            return True
        else:
            return False
    if (porder=="4"):
        if rate<6 and rate>2.9:
            return True
        else:
            return False
    if (porder=="5"):
        if rate<7 and rate>3.9:
            return True
        else:
            return False
    else:
        return False

l1_errors = []
l2_errors = []
linf_errors = []
global_cons_errors = []

def execute_test(grid_1, grid_2, porder, field_type, use_obfet=False, metric=2, two_way_passes=0):
    e = ET.parse('./canga/parameters_template.xml').getroot()
    tree = ET.ElementTree(e)
    
    for item in e.getchildren():
        if (item.attrib['name']=="io"):
            f=item
        if (item.attrib['name']=="remap"):
            g=item
        if (item.attrib['name']=="field"):
            h=item
        if (item.attrib['name']=="two way passes"):
            i=item
        if (item.attrib['name']=="metric"):
            j=item

    
    # have io now
    for item in f.getchildren():
        if (item.attrib['name']=="input file"):
            item.attrib['value']=grid_1;

    # have remap now
    for item in g.getchildren():
        if (item.attrib['name']=="porder"):
            item.attrib['value']=str(porder);
        if (item.attrib['name']=="curvature porder"):
            item.attrib['value']=str(porder);
        if (item.attrib['name']=="obfet"):
            if use_obfet:
                item.attrib['value']="true"
            else:
                item.attrib['value']="false"

    # have field now
    h.attrib['value']=str(field_type)

    # have two way passes now
    i.attrib['value']=str(two_way_passes)

    # have metric now
    j.attrib['value']=str(metric)
 

    tree.write(open('./canga/parameters_lower_1.xml', 'wb'))
    
    # overwrite
    e = ET.parse('./canga/parameters_lower_1.xml').getroot()
    for item in e.getchildren():
        if (item.attrib['name']=="my coloring"):
            item.attrib['value']="33";
        if (item.attrib['name']=="peer coloring"):
            item.attrib['value']="25";

    for item in e.getchildren():
        if (item.attrib['name']=="io"):
            f=item

    # have io now
    for item in f.getchildren():
        if (item.attrib['name']=="input file"):
            item.attrib['value']=grid_2;
    
    tree = ET.ElementTree(e)
    tree.write(open('./canga/parameters_upper_1.xml', 'wb'))
    
    #subprocess.call(['C:\\Temp\\a b c\\Notepad.exe', 'C:\\test.txt'])
    output = subprocess.check_output(["mpirun", "-np", "1", "../bin/cangaIntercomparison.exe","--i=canga/parameters_lower_1.xml","--kokkos-threads=1",":","-np","1","../bin/cangaIntercomparison.exe","--i=canga/parameters_upper_1.xml","--kokkos-threads=1"])
    print output
    if (metric==2):
        m = re.search('(?<=L1 Error: )[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', output)
        l1_errors.append(float(m.group(0)))
        m = re.search('(?<=L2 Error: )[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', output)
        l2_errors.append(float(m.group(0)))
        m = re.search('(?<=LInf Error: )[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', output)
        linf_errors.append(float(m.group(0)))
        return (l1_errors[0],l2_errors[0],linf_errors[0])
    elif (metric==0):
        m = re.search('(?<=Global Conservation Error: )[-]?[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', output)
        global_cons_errors.append(float(m.group(0)))
        return (global_cons_errors[0],)
  
if __name__ == "__main__":
    
    execute_test("mpas_2562.nc","mpas_2562.nc",porder=4,field_type=4,metric=0,use_obfet=0)
    execute_test("mpas_10242.nc","mpas_10242.nc",porder=4,field_type=4,metric=0,use_obfet=0)
    #execute_test("mpas_40962.nc","mpas_40962.nc",porder=4,field_type=4,metric=0,use_obfet=0)
    #execute_test("mpas_163842.nc","mpas_163842.nc",porder=4,field_type=4,metric=0,use_obfet=0)
    
    #execute_test("mpas_2562.nc","mpas_2562.nc",porder=4,field_type=1,metric=2)
    #execute_test("mpas_10242.nc","mpas_10242.nc",porder=4,field_type=1,metric=2)
    #execute_test("mpas_40962.nc","mpas_40962.nc",porder=4,field_type=1,metric=2)
    
    sys.exit(0)
