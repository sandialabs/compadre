import sys
import math
import numpy as np
from netCDF4 import Dataset

# Define bounding coordinates
min_coord, max_coord = -1.0, 1.0

# Number of points in each direction
N = 6

# Generate the 1D spacing
X = np.linspace(min_coord, max_coord, N)
Y = np.linspace(min_coord, max_coord, N)
Z = np.linspace(min_coord, max_coord, N)
# Create a 3D meshgrid
XX, YY, ZZ = np.meshgrid(X, Y, Z)
# Flatten into 1D array
x, y, z = XX.ravel(), YY.ravel(), ZZ.ravel()
# Create arrays to store the normal vectors
nx, ny, nz = 0.0*x, 0.0*y, 0.0*z
# Loop through and set the normal vectors for each point
for i in range(len(x)):
    # Set the components for the normals
    xtemp = (abs(x[i]) == 1.0)*x[i]
    ytemp = (abs(y[i]) == 1.0)*y[i]
    ztemp = (abs(z[i]) == 1.0)*z[i]
    # Normalize vector
    if ( (xtemp == 0.0) and (ytemp == 0.0) and (ztemp == 0.0)):
        nx[i], ny[i], nz[i] = 0.0, 0.0, 0.0
    else:
        nx[i] = xtemp / np.sqrt(xtemp**2 + ytemp**2 + ztemp**2)
        ny[i] = ytemp / np.sqrt(xtemp**2 + ytemp**2 + ztemp**2)
        nz[i] = ztemp / np.sqrt(xtemp**2 + ytemp**2 + ztemp**2)

# Write into file
dataset = Dataset('cube_normal_%d.nc'%N, mode='w', clobber=True, diskless=False,
        persist=False, keepweakref=False, format='NETCDF4')

# Create dimension for number of points
dataset.createDimension('num_points', size=N*N*N)
dataset.createDimension('spatial_dimension', size=3)

# Create variables
dataset.createVariable('x', datatype='d', dimensions=('num_points'), zlib=False, complevel=4,
        shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
        endian='native', least_significant_digit=None, fill_value=None)
dataset.createVariable('y', datatype='d', dimensions=('num_points'), zlib=False, complevel=4,
        shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
        endian='native', least_significant_digit=None, fill_value=None)
dataset.createVariable('z', datatype='d', dimensions=('num_points'), zlib=False, complevel=4,
        shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
        endian='native', least_significant_digit=None, fill_value=None)
dataset.createVariable('nx', datatype='d', dimensions=('num_points'), zlib=False, complevel=4,
        shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
        endian='native', least_significant_digit=None, fill_value=None)
dataset.createVariable('ny', datatype='d', dimensions=('num_points'), zlib=False, complevel=4,
        shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
        endian='native', least_significant_digit=None, fill_value=None)
dataset.createVariable('nz', datatype='d', dimensions=('num_points'), zlib=False, complevel=4,
        shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,
        endian='native', least_significant_digit=None, fill_value=None)

# Copy from numpy array into netCDF4 variables
dataset.variables['x'][:] = x[:]
dataset.variables['y'][:] = y[:]
dataset.variables['z'][:] = z[:]
dataset.variables['nx'][:] = nx[:]
dataset.variables['ny'][:] = ny[:]
dataset.variables['nz'][:] = nz[:]

# Close the file to complete writing
dataset.close()