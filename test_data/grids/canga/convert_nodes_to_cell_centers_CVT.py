import math
import random
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import Delaunay
from scipy.integrate import quad

import sys

def getArea(x,y,z):
    diff_1 = y-x;
    diff_2 = z-x;
    return 0.5*np.linalg.norm(np.cross(diff_1,diff_2))

assert len(sys.argv)>2, "Not enough arguments"
if (len(sys.argv) > 1):
    original_filename = str(sys.argv[1])
if (len(sys.argv) > 2):
    new_filename = str(sys.argv[2])

dataset = Dataset(original_filename, "r", format="NETCDF4")
dimensions = dataset.dimensions
variables = dataset.variables
old_coords = variables['coord']
connect = variables['connect1']

new_midpoint_coords = np.zeros(shape=(dimensions['num_el_in_blk1'].size, 3),dtype='d')
for i in range(dimensions['num_el_in_blk1'].size):
    for j in range(dimensions['num_nod_per_el1'].size):
        for k in range(3):
            new_midpoint_coords[i][k] += old_coords[k][connect[i][j]-1] / float(dimensions['num_nod_per_el1'].size)
#print(new_midpoint_coords)

new_area = np.zeros(shape=(dimensions['num_el_in_blk1'].size,),dtype='d')
for i in range(dimensions['num_el_in_blk1'].size):
    for j in range(dimensions['num_nod_per_el1'].size):
        midpoint = np.reshape(np.array(new_midpoint_coords[i][:]),(3,))
        p1 = np.zeros(shape=(3,),dtype='d')
        p2 = np.zeros(shape=(3,),dtype='d')
        for k in range(3):
            p1[k] = old_coords[k][connect[i][j]-1]
            p2[k] = old_coords[k][connect[i][(j+1)%int(dimensions['num_nod_per_el1'].size)]-1]
        new_area[i] += getArea(midpoint, p1, p2)

f=dataset
dataset2 = Dataset(new_filename, "w", format="NETCDF4")
g=dataset2

for attname in f.ncattrs():
    setattr(g,attname,getattr(f,attname))

# To copy the dimension of the netCDF file
for dimname,dim in f.dimensions.items():
    g.createDimension(dimname,len(dim))

# To copy the variables of the netCDF file
for varname,ncvar in f.variables.items():
    if (varname!="coord"):
        var = g.createVariable(varname,ncvar.dtype,ncvar.dimensions)
        #Proceed to copy the variable attributes
        #for attname in ncvar.ncattrs():
        #   setattr(var,attname,getattr(ncvar,attname))
        var[:] = ncvar[:]
#    else:
#        g.createVariable('coord', datatype='d', dimensions=('num_el_in_blk1','num_dim'), zlib=False, complevel=4,\
#                               shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
#                               endian='native', least_significant_digit=None, fill_value=None)
#        #for attname in ncvar.ncattrs():
#        #   setattr(var,attname,getattr(ncvar,attname))
#        g.variables['coord'][:]=new_midpoint_coords
g.createVariable('x', datatype='d', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                       endian='native', least_significant_digit=None, fill_value=None)
g.variables['x'][:]=new_midpoint_coords[:,0]
g.createVariable('y', datatype='d', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                       endian='native', least_significant_digit=None, fill_value=None)
g.variables['y'][:]=new_midpoint_coords[:,1]
g.createVariable('z', datatype='d', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                       endian='native', least_significant_digit=None, fill_value=None)
g.variables['z'][:]=new_midpoint_coords[:,2]

g.createVariable('area', datatype='d', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                       endian='native', least_significant_digit=None, fill_value=None)
g.variables['area'][:]=new_area
print("AREA:" + str(np.sum(new_area)))

dataset2.close()
dataset.close()
