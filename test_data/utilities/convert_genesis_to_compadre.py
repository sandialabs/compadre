import os
import sys
import numpy as np
from netCDF4 import Dataset
import argparse
from numba import jit

@jit(nopython=True,parallel=False)
def getFlattenCellVertices(i, new_data, coords, connect, nod_size, dim):
    for j in range(nod_size):
        for k in range(dim):
            new_data[dim*j+k] = coords[connect[i][j]-1,k]

@jit(nopython=True,parallel=False)
def getNewMidpoints(i, new_data, coords, connect, nod_size):
    for j in range(nod_size):
        for k in range(3):
            new_data[k] += coords[connect[i,j]-1, k] / float(nod_size)

parser = argparse.ArgumentParser(description='reads in genesis files\
    default for coordinates is that they are stored in field variables \'x\', \'y\', and \'z\'\
    (can be modified with command line argumentstake diff of field on file1 and file2 and store in new-file')

parser.add_argument('--file-in', dest='file_in', type=str, help='file in name')
parser.add_argument('--file-out', dest='file_out', type=str, help='file out name')
parser.add_argument('--xyz-type', dest='xyz_type', type=str, default='separate', help='{\'separate\',\'joint\'}')
parser.add_argument('--xyz-name', dest='xyz_name', type=str, default='x', help='xyz_name (if xyz_joint chosen), or x name')
parser.add_argument('--y-name', dest='y_name', type=str, default='y', help='y name (if xyz_separate chosen) or unused')
parser.add_argument('--z-name', dest='z_name', type=str, default='z', help='z name (if xyz_separate chosen) or unused')
parser.add_argument('--cell-to-vertex-field-name', dest='cell_to_vertex_field_name', type=str, default='connect1', help='name for dimension of cells')
parser.add_argument('--dimension-nodes-per-element-name', dest='dimension_nodes_per_element_name', type=str, default='num_nod_per_el1', help='name for dimension of nodes per element')
parser.add_argument('--dim', dest='dim', type=int, default=3, help='spatial dimension')
args = parser.parse_args()

file_in = args.file_in
file_out = args.file_out
xyz_types = ['separate','joint']
xyz_type = xyz_types.index(args.xyz_type)
x_name = args.xyz_name
y_name = args.y_name
z_name = args.z_name

dataset = Dataset(file_in, "r", format="NETCDF4")
dimensions = dataset.dimensions
variables = dataset.variables

# get dimensions of coordinates
n_xyz = 0
for variable_name in variables:
    if (str(variable_name)==x_name):
        assert len(variables[variable_name].shape)<=2, "Doesn't support fields with more than two axis."
        n_xyz = variables[variable_name].shape[0]

# copy coordinates
coords = np.zeros(shape=(n_xyz, 3), dtype='f8')
coord_x_found = False
coord_y_found = False
coord_z_found = False
for variable_name in variables:
    if (str(variable_name)==x_name):
        if (xyz_type==0):
            coords[:,0] = variables[variable_name][:]
            coord_x_found = True
        else:
            coords[:,:] = variables[variable_name][:,:]
            coord_x_found = True
            coord_y_found = True
            coord_z_found = True
    if (str(variable_name)==y_name):
        if (xyz_type==0):
            coords[:,1] = variables[variable_name][:]
            coord_y_found = True
    if (str(variable_name)==z_name):
        if (xyz_type==0):
            coords[:,2] = variables[variable_name][:]
            coord_z_found = True

assert coord_x_found, "Field for --xyz-name, %s, was not found."%x_name
if (args.dim>1):
    assert coord_y_found, "Field for --y-name, %s, was not found."%y_name
if (args.dim>2):
    assert coord_z_found, "Field for --z-name, %s, was not found."%z_name

#print(coords)
connect = np.array(variables[args.cell_to_vertex_field_name])
el_size = connect.shape[0]
nod_size = int(dimensions[args.dimension_nodes_per_element_name].size)

# get cell centers from coordinates
midpoints = np.zeros(shape=(el_size,3), dtype='f8')
for i in range(el_size):
    this_new_midpoint_coords = np.zeros(3)
    getNewMidpoints(i, this_new_midpoint_coords, coords, connect, nod_size)
    midpoints[i,:] = this_new_midpoint_coords
print(midpoints)

vertex_coordinates = np.zeros(shape=(el_size, args.dim*nod_size),dtype='f8')
for i in range(el_size):
    this_extra_data = np.zeros(args.dim*nod_size)
    getFlattenCellVertices(i, this_extra_data, coords, connect, nod_size, args.dim)
    vertex_coordinates[i,:] = this_extra_data
print(vertex_coordinates)

# only works for simplices
adjacent_elements = -np.ones(shape=(el_size, args.dim+1),dtype='i8') 

new_ID = np.arange(el_size)



