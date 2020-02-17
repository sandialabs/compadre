import os
import sys
import vtk
import numpy as np
from netCDF4 import Dataset

# reads in pvtp, pvtu, or vtk
# converts points to or 0) xyz_separate('x':x, 'y':y, 'z':z) or 1) xyz_joint('xyz':(x,y,z) 
# all other fields copied exactly as they are to netcdf (.nc)

assert len(sys.argv) > 2, "Not enough input arguments"

# argument 1: (string) file in name
# argument 2: (string) file out name
# argument 3: (int) xyz_type
# argument 4: (string) xyz_name (if xyz_joint chosen), or x name 
# argument 5: (string) y name (if xyz_separate chosen) or unused
# argument 6: (string) z name (if xyz_separate chosen) or unused

file_in = ""
file_out = ""
xyz_type = 0
x_name = "x"
y_name = "y"
z_name = "z"

if (len(sys.argv) > 1):
    file_in = sys.argv[1]
if (len(sys.argv) > 2):
    file_out = sys.argv[2]
if (len(sys.argv) > 3):
    xyz_type = int(sys.argv[3])
if (len(sys.argv) > 4):
    x_name = sys.argv[4]
if (len(sys.argv) > 5):
    y_name = sys.argv[5]
if (len(sys.argv) > 6):
    z_name = sys.argv[6]

# get ending of input filename
filename, file_extension = os.path.splitext(file_in)

reader = None
if (file_extension==".pvtp"):
    reader = vtk.vtkXMLPPolyDataReader()
elif (file_extension==".pvtu"):
    reader = vtk.vtkXMLPUnstructuredGridReader()
elif (file_extension==".vtp"):
    reader = vtk.vtkXMLPolyDataReader()
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
elif (file_extension==".vtu"):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
elif (file_extension==".vtk"):
    reader = vtk.vtkPolyDataReader()
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
else:
    assert False, "file input extension name not recognized"
reader.SetFileName(file_in)
reader.Update()

data= reader.GetOutput() # contains point data
pd = data.GetPointData() # contains all other variables data

# number of coordinates
n_xyz = data.GetNumberOfPoints()
# copy to numpy array
coords = np.zeros(shape=(n_xyz, 3), dtype='f8')
for i in range(n_xyz):
    xyz_val = data.GetPoint(i)
    coords[i,:] = xyz_val

# get all field data
fields = list()
for i in range(pd.GetNumberOfArrays()):
    rows = pd.GetArray(i).GetNumberOfTuples()
    cols = pd.GetArray(i).GetNumberOfComponents()
    data = None
    if (str(type(pd.GetArray(i))) == '<class \'vtkCommonCorePython.vtkDoubleArray\'>'):
        data = np.zeros(shape=(rows,cols), dtype='f8')
    elif (str(type(pd.GetArray(i))) == '<class \'vtkCommonCorePython.vtkFloatArray\'>'):
        data = np.zeros(shape=(rows,cols), dtype='f8')
    elif (str(type(pd.GetArray(i))) == '<class \'vtkCommonCorePython.vtkIntArray\'>'):
        data = np.zeros(shape=(rows,cols), dtype='i4')
    elif (str(type(pd.GetArray(i))) == '<class \'vtkCommonCorePython.vtkLongArray\'>' or \
            str(type(pd.GetArray(i))) == '<class \'vtkCommonCorePython.vtkLongLongArray\'>'):
        data = np.zeros(shape=(rows,cols), dtype='i8')
    else:
        assert False, "Encountered field of a data type(%s) in VTK(%s) that is not supported." % (type(pd.GetArray(i)), file_in)
    for j in range(rows):
        data[j,:] = pd.GetArray(i).GetTuple(j)
    fields.append([pd.GetArrayName(i), data])
    if (xyz_type==0): # separate
        assert fields[i][0]!=x_name and fields[i][0]!=y_name and fields[i][0]!=z_name, \
            "Field(%s) shares a name with coordinate variable name(%s, %s, or %s)."%(fields[i][0], x_name, y_name, z_name)
    elif (xyz_type==1): # joint
        assert fields[i][0]!=x_name, "Field(%s) shares a name with coordinate variable name(%s)."%(fields[i][0], x_name)
    print(fields[i][0])


# all point data now collected (coordinates + fields from VTK)


# write solution to NetCDF
dataset = Dataset(file_out, mode="w", clobber=True, diskless=False,\
                   persist=False, keepweakref=False, format='NETCDF4')
dataset.createDimension('num_coords', size=coords.shape[0])
if (xyz_type==1): # joint xyz
    dataset.createDimension('coords_dim', size=coords.shape[1])


# creates dimension for field (either first or second dimension), if needed
# if it matches same as # of coordinates, then uses that dimension
for field in fields:
    if (field[1].shape[0]!=coords.shape[0]):
        dataset.createDimension('%s_dim_1'%(field[0],), size=field[1].shape[0])
    if (field[1].shape[1]!=coords.shape[0]):
        if (field[1].shape[1] > 1):
            dataset.createDimension('%s_dim_2'%(field[0],), size=field[1].shape[1])

# create coordinates variable
dimensions = list()
if (xyz_type==0): # separate
    dimensions = ['num_coords',]
    dataset.createVariable(x_name, datatype='f8', dimensions=tuple(dimensions), zlib=True, complevel=8,\
                      shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                      endian='native', least_significant_digit=None, fill_value=None)
    dataset.createVariable(y_name, datatype='f8', dimensions=tuple(dimensions), zlib=True, complevel=8,\
                      shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                      endian='native', least_significant_digit=None, fill_value=None)
    dataset.createVariable(z_name, datatype='f8', dimensions=tuple(dimensions), zlib=True, complevel=8,\
                      shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                      endian='native', least_significant_digit=None, fill_value=None)
elif (xyz_type==1): # joint
    dimensions = ['num_coords', 'coords_dim']
    dataset.createVariable(x_name, datatype='f8', dimensions=tuple(dimensions), zlib=True, complevel=8,\
                      shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                      endian='native', least_significant_digit=None, fill_value=None)

# create field and match to dimensions
for field in fields:
    dtype = ''
    if (str(field[1].dtype)=='float64'): dtype = 'f8'
    if (str(field[1].dtype)=='int32'): dtype = 'i4'
    if (str(field[1].dtype)=='int64'): dtype = 'i8'

    dimensions = ['',]
    if (field[1].shape[0]!=coords.shape[0]):
        dimensions[0] = '%s_dim_1'%(field[0],)
    else:
        dimensions[0] = 'num_coords'

    if (field[1].shape[1]>1):
        if (field[1].shape[1]!=coords.shape[0]):
            dimensions.append('%s_dim_2'%(field[0],))
        else:
            dimensions.append('num_coords')

    dataset.createVariable(field[0], datatype=dtype, dimensions=tuple(dimensions), zlib=True, complevel=8,\
                      shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                      endian='native', least_significant_digit=None, fill_value=None)

# fill coordinates with values
if (xyz_type==0): # separate
    dataset.variables[x_name][:] = coords[:,0]
    dataset.variables[y_name][:] = coords[:,1]
    dataset.variables[z_name][:] = coords[:,2]
elif (xyz_type==1): # joint
    dataset.variables[x_name][:,:] = coords[:,:]

# fill fields with values
for field in fields:
    if (len(dataset.variables[field[0]].shape)>1):
        dataset.variables[field[0]][:,:] = field[1][:,:]
    else:
        dataset.variables[field[0]][:] = field[1][:]

dataset.close()
















