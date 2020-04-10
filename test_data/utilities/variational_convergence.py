import xml.etree.ElementTree as ET
import subprocess
import os
import re
import math
import sys
import argparse

parser = argparse.ArgumentParser(description='Run convergence tests and calculate convergence rate.')

parser.add_argument('--num-meshes', dest='num_meshes', type=int, default=3, nargs='?', help='number of meshes')
parser.add_argument('--order', dest='order', type=int, nargs='?', default=2, help='polynomial order for basis')
parser.add_argument('--pressure-order', dest='pressure_order', type=int, nargs='?', default=-1, help='polynomial order for pressure basis')

parser.add_argument('--shear', dest='shear', type=float, nargs='?', default=1.0, help='shear modulus')
parser.add_argument('--lambda', dest='lambda_lame', type=float, nargs='?', default=1.0, help='lambda coefficient')
parser.add_argument('--reaction', dest='reaction', type=float, nargs='?', default=1.0, help='reaction coefficient')
parser.add_argument('--diffusion', dest='diffusion', type=float, nargs='?', default=1.0, help='diffusion coefficient')
parser.add_argument('--penalty', dest='penalty', type=float, nargs='?', default=1.0, help='penalty coefficient')
parser.add_argument('--size', dest='size', type=float, nargs='?', default=1.0, help='first mesh search size (halfed each refinement)')
parser.add_argument('--rate-tol', dest='rate_tol', type=float, nargs='?', default=0.5, help='tolerance for convergence')

parser.add_argument('--solution', dest='solution', type=str, nargs='?', default='polynomial', help='solution type')
parser.add_argument('--pressure-solution', dest='pressure_solution', type=str, nargs='?', default='polynomial', help='pressure solution type')
parser.add_argument('--pressure-null-space', dest='pressure_null_space', type=str, nargs='?', default='pinning', help='treatment of null space in pressure {"pinning", "lm", "none}')
parser.add_argument('--operator', dest='operator', type=str, nargs='?', default='rd', help='operator for PDE solve')
parser.add_argument('--convergence-type', dest='convergence_type', type=str, nargs='?', default='rate', help='type of convergence to test')

parser.add_argument('--assert-rate', dest='assert_rate', type=str, nargs='?', default='True', help='whether to assert rate is optimal')

args = parser.parse_args()

if args.pressure_order<0:
    args.pressure_order = args.order-1 # Taylor-Hood style velocity+pressure pair


file_names = ["dg_%d.nc"%num for num in range(args.num_meshes)]
error_types=['vel. l2','vel. h1','vel. jp','vel. sum','pr. l2']
all_errors = [list(), list(), list(), list(), list(), list(), list()]#list() * len(error_types)

for key2, fname in enumerate(file_names):
    e = ET.parse('../../test_data/parameter_lists/reactiondiffusion/parameters_template.xml').getroot()
    tree = ET.ElementTree(e)
    size_str = str(args.size/float(pow(2,key2)))
    print(size_str)
    
    for item in e.getchildren():
        if (item.attrib['name']=="io"):
            f=item
        if (item.attrib['name']=="neighborhood"):
            n=item
        if (item.attrib['name']=="physics"):
            p=item
        if (item.attrib['name']=="remap"):
            r=item
        if (item.attrib['name']=="solver"):
            s=item
    
    for item in f.getchildren():
        if (item.attrib['name']=="input file"):
            item.attrib['value']=fname

    for item in n.getchildren():
        if (item.attrib['name']=="size"):
            item.attrib['value']=size_str

    for item in p.getchildren():
        if (item.attrib['name']=="solution"):
            item.attrib['value']=args.solution
        if (item.attrib['name']=="pressure solution"):
            item.attrib['value']=args.pressure_solution
        if (item.attrib['name']=="operator"):
            item.attrib['value']=args.operator
        if (item.attrib['name']=="reaction"):
            item.attrib['value']=str(args.reaction)
        if (item.attrib['name']=="diffusion"):
            item.attrib['value']=str(args.diffusion)
        if (item.attrib['name']=="shear"):
            item.attrib['value']=str(args.shear)
        if (item.attrib['name']=="lambda"):
            item.attrib['value']=str(args.lambda_lame)
        if (item.attrib['name']=="penalty"):
            item.attrib['value']=str(args.penalty)

    for item in r.getchildren():
        if (item.attrib['name']=="porder"):
            item.attrib['value']=str(args.order)
        if (item.attrib['name']=="pressure porder"):
            item.attrib['value']=str(args.pressure_order)

    for item in s.getchildren():
        if (item.attrib['name']=="pressure null space"):
            item.attrib['value']=str(args.pressure_null_space)
    
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
        if (args.operator != 'l2'):
            m = re.search('(?<=H1: )[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', output)
            all_errors[1].append(float(m.group(0)))
            m = re.search('(?<=Ju: )[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', output)
            all_errors[2].append(float(m.group(0)))
            m = re.search('(?<=Global Norm: )[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', output)
            all_errors[3].append(float(m.group(0)))
        if (args.operator.lower() in ['st', 'mix_le']):
            m = re.search('(?<=Pressure L2: )[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', output)
            all_errors[4].append(float(m.group(0)))
    
print(all_errors)
for key, errors in enumerate(all_errors):
    rate = 0
    last_error = 0
    if (args.num_meshes>1 and len(errors)>0):
        print("\n\nerror rates: type:%s\n============="%(error_types[key],))
        for i in range(1,len(errors)):
            if (errors[i]!=0):
                rate = math.log(errors[i]/errors[i-1])/math.log(.5)
                last_error = errors[i]
                print(str(rate) + ", " + str(errors[i]) + ", " + str(errors[i-1]))
            else:
                print("NaN - Division by zero")
    else:
        break

    if (args.convergence_type.lower()=="exact" and args.assert_rate.lower()=="true"):
        if (abs(args.rate_tol-last_error)>args.rate_tol):
            assert False, "Last calculated error (%f) more than %f from exact solution." % (last_error, args.rate_tol,)

    elif (args.convergence_type.lower()=="rate" and args.assert_rate.lower()=="true"):
        rate_adjustment = 0
        if ('l2' in error_types[key] and args.operator.lower()!="mix_le"):
            rate_adjustment = 1
        if (abs(args.order+rate_adjustment-rate)>args.rate_tol and rate<args.order):
            assert False, "Last calculated rate (%f) more than %f from theoretical optimal rate." % (rate, args.rate_tol,)


sys.exit(0)
