# script used to add an ID, add a smooth pointwise sampled variable,
# convert lat/lon to x,y,z, and write a field called "extra data" 
# which contains the (x,y,z) values for each vertex of a cell

import math
import random
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import Delaunay
from scipy.integrate import quad
import argparse

import sys
from numba import jit

def convert(original_filename, new_filename):
    
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
    old_lat = variables['lat']
    old_lon = variables['lon']
    
    @jit(nopython=True,parallel=False)
    def getNewMidpoints(i, new_data, old_coords, connect, el_size, nod_size):
        for j in range(nod_size):
            for k in range(3):
                new_data[k] += old_coords[k][connect[i][j]-1] / float(nod_size)
    
    @jit(nopython=True,parallel=False)
    def getFlattenCellVertices(i, new_data, old_coords, connect, el_size, nod_size):
        for j in range(nod_size):
            for k in range(3):
                new_data[3*j+k] = old_coords[k][connect[i][j]-1]
    
    @jit(nopython=True,parallel=False)
    def transformLatLon(new_data, old_lat, old_lon, in_degrees=True):
        #if in_degrees:
        lat = float(old_lat * np.pi) / 180.0
        lon = float(old_lon * np.pi) / 180.0
        #else:
        #    lat = old_lat
        #    lon = old_lon
    
        if (lat > 0.5 * np.pi): lat = 0.5 * np.pi
        if (lat < -0.5 * np.pi): lat = -0.5 * np.pi
    
        # should be consistent with computeSphericalCartesianTransforms.py from metrics
        new_data[0] = np.cos(lon) * np.cos(lat)
        new_data[1] = np.sin(lon) * np.cos(lat)
        new_data[2] = np.sin(lat)
    
    @jit(nopython=True,parallel=False)
    def getSmoothVal(i, new_data, old_coords):
        x = old_coords[i,0]
        y = old_coords[i,1]
        z = old_coords[i,2]
        new_data[0] = np.sin(x)*np.sin(y)*np.sin(z)
    
    el_size = dimensions['num_el_in_blk1'].size
    nod_size = dimensions['num_nod_per_el1'].size
    np_old_coords = np.array(old_coords)
    np_connect = np.array(connect)
    
    extra_data = np.zeros(shape=(el_size, 3*nod_size),dtype='f8')
    for i in range(el_size):
        this_extra_data = np.zeros(3*nod_size, dtype='f8')
        getFlattenCellVertices(i, this_extra_data, np_old_coords, np_connect, el_size, nod_size)
        extra_data[i,:] = this_extra_data
    
    alt_new_midpoint_coords = np.zeros(shape=(el_size, 3),dtype='f8')
    for i in range(el_size):
        this_new_midpoint_coords = np.zeros(3, dtype='f8')
        transformLatLon(this_new_midpoint_coords, old_lat[i], old_lon[i])
        alt_new_midpoint_coords[i,:] = this_new_midpoint_coords
    
    # check differences 
    #tol = 1e-1
    #for i in range(el_size):
    #    diff = 0
    #    for k in range(3):
    #        diff += new_midpoint_coords[i,k] - alt_new_midpoint_coords[i,k]
    #    diff = math.sqrt(abs(diff))
    #    if (diff > tol):
    #        print("DIFF>TOL %d: %f, %f" % (i, diff, tol))
    
    smooth = np.zeros(shape=(el_size, 1),dtype='f8')
    for i in range(el_size):
        this_smooth_val = np.zeros(1, dtype='f8')
        getSmoothVal(i, this_smooth_val, alt_new_midpoint_coords)
        smooth[i,0] = this_smooth_val[0]
    
    #vertices_per_cell = dimensions['num_nod_per_el1']
    #extra_data = np.zeros(shape=(dimensions['num_el_in_blk1'].size, vertices_per_cell),dtype='d')
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
    
    # make extra data dim
    g.createDimension("extra_data_dim",3*nod_size)
    
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
    g.variables['x'][:]=alt_new_midpoint_coords[:,0]
    g.createVariable('y', datatype='f8', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)
    g.variables['y'][:]=alt_new_midpoint_coords[:,1]
    g.createVariable('z', datatype='f8', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)
    g.variables['z'][:]=alt_new_midpoint_coords[:,2]
    
    g.createVariable('ID', datatype='i8', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)
    g.variables['ID'][:]=new_ID
    
    g.createVariable('Smooth', datatype='f8', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)
    g.variables['Smooth'][:]=smooth
    
    g.createVariable('extra_data', datatype='f8', dimensions=('num_el_in_blk1','extra_data_dim'), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)
    g.variables['extra_data'][:]=extra_data
    
    dataset2.close()
    dataset.close()

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='convert files by adding cell centroids and ID (for cubed-sphere)')
    parser.add_argument('--original-file', dest='original_file', type=str, help='original file name')
    parser.add_argument('--new-file', dest='new_file', type=str, help='new file name')
    args = parser.parse_args()
    convert(args.original_file, args.new_file)
