# script used to get cell centers and add an ID

import math
import random
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import Delaunay
from scipy.integrate import quad

import sys
from numba import jit


assert len(sys.argv)>2, "Not enough arguments"
if (len(sys.argv) > 1):
    original_filename = str(sys.argv[1])
if (len(sys.argv) > 2):
    new_filename = str(sys.argv[2])

dataset = Dataset(original_filename, "r", format="NETCDF4")
dimensions = dataset.dimensions
variables = dataset.variables

#read_coords = variables['coord'] # with reversed xy
#old_coords = np.zeros(shape=read_coords.shape, dtype='d')
#for i in range(dimensions['num_nodes'].size):
#    old_coords[0][i] = read_coords[0][i]
#    old_coords[1][i] = read_coords[1][i]
#    old_coords[2][i] = read_coords[2][i]

old_coords = variables['coord']
connect = variables['connect1']

@jit(nopython=True,parallel=False)
def getNewMidpoints(i, new_data, old_coords, connect, el_size, nod_size):
    for j in range(nod_size):
        for k in range(3):
            new_data[k] += old_coords[k][connect[i][j]-1] / float(nod_size)

el_size = dimensions['num_el_in_blk1'].size
nod_size = dimensions['num_nod_per_el1'].size
new_midpoint_coords = np.zeros(shape=(el_size, 3),dtype='d')
np_old_coords = np.array(old_coords)
np_connect = np.array(connect)

for i in range(el_size):
    this_new_midpoint_coords = np.zeros(3)
    getNewMidpoints(i, this_new_midpoint_coords, np_old_coords, np_connect, el_size, nod_size)
    new_midpoint_coords[i,:] = this_new_midpoint_coords

#new_midpoint_coords = np.zeros(shape=(dimensions['num_el_in_blk1'].size, 3),dtype='d')
#for i in range(dimensions['num_el_in_blk1'].size):
#    num_nodes = dimensions['num_nod_per_el1'].size
#    for j in range(num_nodes):
#        for k in range(3):
#            new_midpoint_coords[i,k] += old_coords[k,connect[i,j]-1]

new_ID = np.arange(dimensions['num_el_in_blk1'].size)

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
g.createVariable('x', datatype='f8', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                       endian='native', least_significant_digit=None, fill_value=None)
g.variables['x'][:]=new_midpoint_coords[:,0]
g.createVariable('y', datatype='f8', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                       endian='native', least_significant_digit=None, fill_value=None)
g.variables['y'][:]=new_midpoint_coords[:,1]
g.createVariable('z', datatype='f8', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                       endian='native', least_significant_digit=None, fill_value=None)
g.variables['z'][:]=new_midpoint_coords[:,2]

g.createVariable('ID', datatype='i8', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                       endian='native', least_significant_digit=None, fill_value=None)
g.variables['ID'][:]=new_ID

dataset2.close()
dataset.close()
