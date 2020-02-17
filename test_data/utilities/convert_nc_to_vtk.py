import os
import sys
import vtk
import numpy as np
from netCDF4 import Dataset

# reads in netcdf files
# all other fields copied exactly as they are to pvtp, pvtu, or vtk
# default for coordinates is that they are stored in field variables 'x', 'y', and 'z' 
# (can be modified with command line arguments)

assert len(sys.argv) > 2, "Not enough input arguments"

# choose how coordinates are layed out with xyz_type of 0) xyz_separate('x':x, 'y':y, 'z':z) or 1) xyz_joint('xyz':(x,y,z) 
# argument 1: (string) file in name
# argument 2: (string) file out name
# argument 3: (bool) save to binary?
# argument 3: (int) xyz_type
# argument 4: (string) xyz_name (if xyz_joint chosen), or x name 
# argument 5: (string) y name (if xyz_separate chosen) or unused
# argument 6: (string) z name (if xyz_separate chosen) or unused


file_in = ""
file_out = ""
to_binary = True
xyz_type = 0
x_name = "x"
y_name = "y"
z_name = "z"

if (len(sys.argv) > 1):
    file_in = sys.argv[1]
if (len(sys.argv) > 2):
    file_out = sys.argv[2]
if (len(sys.argv) > 3):
    to_binary = bool(sys.argv[3])
if (len(sys.argv) > 4):
    xyz_type = int(sys.argv[4])
if (len(sys.argv) > 5):
    x_name = sys.argv[5]
if (len(sys.argv) > 6):
    y_name = sys.argv[6]
if (len(sys.argv) > 7):
    z_name = sys.argv[7]


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
for variable_name in variables:
    if (str(variable_name)==x_name):
        if (xyz_type==0):
            coords[:,0] = variables[variable_name][:]
        else:
            coords[:,:] = variables[variable_name][:,:]
    if (str(variable_name)==y_name):
        if (xyz_type==0):
            coords[:,1] = variables[variable_name][:]
    if (str(variable_name)==z_name):
        if (xyz_type==0):
            coords[:,2] = variables[variable_name][:]


fields = dict()
for variable_name in variables:
    if (xyz_type==0):
        if (str(variable_name)==x_name or str(variable_name)==y_name or str(variable_name)==z_name):
            continue
    else:
        if (str(variable_name)==x_name):
            continue
    values = np.ndarray(variables[variable_name].shape, dtype=variables[variable_name].dtype)
    if (variables[variable_name].ndim==1):
        values[:] = variables[variable_name][:]
    elif (variables[variable_name].ndim==2):
        values[:,:] = variables[variable_name][:,:]
    fields[str(variable_name)] = values

dataset.close()


points = vtk.vtkPoints()
points.SetNumberOfPoints(coords.shape[0])
for i in range(coords.shape[0]):
    points.InsertPoint(i, coords[i,0], coords[i,1], coords[i,2])
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
pointdata = polydata.GetPointData()

def np_2_vtk_array(np_array):

    vtk_array = None
    if (str(np_array.dtype)=="float64" or str(np_array.dtype)=="float32"):
        vtk_array = vtk.vtkDoubleArray()
    elif (str(np_array.dtype)=="int32"):
        vtk_array = vtk.vtkIntArray()
    elif (str(np_array.dtype)=="int64"):
        vtk_array = vtk.vtkLongLongArray()
    else:
        assert False, "Invalid dtype for np_array given to np_2_vtk_array."
    
    if np_array.ndim < 2:
      vtk_array.SetNumberOfComponents(1)
      vtk_array.SetNumberOfTuples(np_array.size)
    else:
      vtk_array.SetNumberOfComponents(np_array.shape[1])
      vtk_array.SetNumberOfTuples(np_array.shape[0])

    vtk_array.SetVoidArray(np_array, np_array.size, 1)
    return vtk_array

for field in fields:
    vtk_array = np_2_vtk_array(fields[field])
    vtk_array.SetName(field)
    pointdata.AddArray(vtk_array)

# get ending of input filename
filename, file_extension = os.path.splitext(file_out)
writer = None
if (file_extension==".pvtp"):
    writer = vtk.vtkXMLPPolyDataWriter()
elif (file_extension==".pvtu"):
    writer = vtk.vtkXMLPUnstructuredGridWriter()
elif (file_extension==".vtp"):
    writer = vtk.vtkXMLPolyDataWriter()
elif (file_extension==".vtu"):
    writer = vtk.vtkXMLUnstructuredGridWriter()
elif (file_extension==".vtk"):
    writer = vtk.vtkPolyDataWriter()
else:
    assert False, "file input extension name not recognized"

polydata.Modified()

writer.SetInputData(polydata)
writer.SetFileName(file_out)
#if to_binary:
#    writer.SetFileTypeToBinary()
writer.Update()
writer.Write()
