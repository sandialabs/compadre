import math
import random
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import Delaunay
from scipy.integrate import quad
import argparse

import sys
import time
from numba import jit

#@jit(nopython=True,parallel=True)
#def getArea(x,y,z):
#    diff_1 = y-x;
#    diff_2 = z-x;
#    t0 = diff_1[1]*diff_2[2]-diff_2[1]*diff_1[2]
#    t1 = -(diff_1[0]*diff_2[2]-diff_2[0]*diff_1[2])
#    t2 = diff_1[0]*diff_2[1]-diff_2[0]*diff_1[1]
#    return 0.5*np.sqrt(t0*t0+t1*t1+t2*t2)
##    return 0.5*np.linalg.norm(np.cross(diff_1,diff_2))


def convert(original_filename, new_filename):
    
    dataset = Dataset(original_filename, "r", format="NETCDF4")
    dimensions = dataset.dimensions
    variables = dataset.variables
    old_coords = variables['coord']
    connect = variables['connect1']
    
    t0 = time.time()
    
    @jit(nopython=True,parallel=False)
    def getNewMidpoints(i, new_data, old_coords, connect, el_size, nod_size):
        #new_midpoint_coords = np.zeros(shape=(el_size, 3),dtype='d')
        #for i in range(el_size):
        #new_midpoint_coords = np.zeros(3)
        for j in range(nod_size):
            for k in range(3):
                #new_midpoint_coords[i,k] += old_coords[k,connect[i,j]-1] / float(nod_size)
                #new_midpoint_coords[i][k] += old_coords[k][connect[i][j]-1] / float(nod_size)
                new_data[k] += old_coords[k][connect[i][j]-1] / float(nod_size)
    
        # norm 1 puts the point on the sphere
        norm = 0
        for k in range(3):
            norm += new_data[k]*new_data[k]
        norm = math.sqrt(norm)
        new_data = new_data / norm
    
        #return new_midpoint_coords
    
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
    
    el_size = dimensions['num_el_in_blk1'].size
    nod_size = dimensions['num_nod_per_el1'].size
    np_old_coords = np.array(old_coords)
    old_lat = variables['lat']
    old_lon = variables['lon']
    np_connect = np.array(connect)
    print(np_old_coords)
    
    #new_midpoint_coords = np.zeros(shape=(el_size, 3),dtype='f8')
    #for i in range(el_size):
    #    this_new_midpoint_coords = np.zeros(3)
    #    getNewMidpoints(i, this_new_midpoint_coords, np_old_coords, np_connect, el_size, nod_size)
    #    new_midpoint_coords[i,:] = this_new_midpoint_coords
    alt_new_midpoint_coords = np.zeros(shape=(el_size, 3),dtype='f8')
    for i in range(el_size):
        this_new_midpoint_coords = np.zeros(3, dtype='f8')
        transformLatLon(this_new_midpoint_coords, old_lat[i], old_lon[i])
        alt_new_midpoint_coords[i,:] = this_new_midpoint_coords
    
    extra_data = np.zeros(shape=(el_size, 3*nod_size),dtype='f8')
    for i in range(el_size):
        this_extra_data = np.zeros(3*nod_size)
        getFlattenCellVertices(i, this_extra_data, np_old_coords, np_connect, el_size, nod_size)
        extra_data[i,:] = this_extra_data
    
    #print(new_midpoint_coords)
    t1 = time.time()
    total = t1-t0
    print(str(total)+"coords")
    #new_area = np.zeros(shape=(el_size,),dtype='d')
    #temp_p1 = np.zeros(shape=(el_size,3),dtype='d')
    #temp_p2 = np.zeros(shape=(el_size,3),dtype='d')
    #
    #@jit(nopython=True,parallel=True)
    #def getNewArea(i, p1, p2, new_midpoint_coords, old_coords, connect):
    #    this_new_area = 0
    #    for j in range(nod_size):
    #        #midpoint = np.reshape(np.array(new_midpoint_coords[i][:]),(3,))
    #        #p1 = np.zeros(shape=(3,),dtype='d')
    #        #p2 = np.zeros(shape=(3,),dtype='d')
    #        for k in range(3):
    #            p1[i,k] = old_coords[k][connect[i][j]-1]
    #            p2[i,k] = old_coords[k][connect[i][(j+1)%int(nod_size)]-1]
    #        #this_new_area += getArea(midpoint, p1[:,k], p2[:,k])
    #        this_new_area += getArea(new_midpoint_coords[i,:], p1[i,:], p2[i,:])
    #    return this_new_area
    #
    #for i in range(el_size):
    #    this_new_area = getNewArea(i, temp_p1, temp_p2, new_midpoint_coords, np_old_coords, np_connect)
    #    new_area[i] = this_new_area
    #
    #t2 = time.time()
    #total = t2-t1
    #print(str(total)+"areas")
    
    new_ID = np.arange(dimensions['num_el_in_blk1'].size)
    
    f=dataset
    dataset2 = Dataset(new_filename, "w", format="NETCDF4")
    g=dataset2
    
    for attname in f.ncattrs():
        setattr(g,attname,getattr(f,attname))
    
    # To copy the dimension of the netCDF file
    for dimname,dim in f.dimensions.items():
        g.createDimension(dimname,len(dim))
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
    
    g.createVariable('extra_data', datatype='f8', dimensions=('num_el_in_blk1','extra_data_dim'), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)
    g.variables['extra_data'][:]=extra_data
    
    #g.createVariable('area', datatype='d', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
    #                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
    #                       endian='native', least_significant_digit=None, fill_value=None)
    #g.variables['area'][:]=new_area
    #print("AREA:" + str(np.sum(new_area)))
    
    dataset2.close()
    dataset.close()

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='convert files by adding cell centroids and ID (for ICOD)')
    parser.add_argument('--original-file', dest='original_file', type=str, help='original file name')
    parser.add_argument('--new-file', dest='new_file', type=str, help='new file name')
    args = parser.parse_args()
    convert(args.original_file, args.new_file)
