import os
import sys
import vtk
import numpy as np
from netCDF4 import Dataset
import argparse

parser = argparse.ArgumentParser(description='reads in netcdf files\
    all other fields copied exactly as they are to pvtp, pvtu, or vtk\
    default for coordinates is that they are stored in field variables \'x\', \'y\', and \'z\'\
    (can be modified with command line argumentstake diff of field on file1 and file2 and store in new-file')

parser.add_argument('--file-in', dest='file_in', type=str, help='file in name')
parser.add_argument('--file-out', dest='file_out', type=str, help='file out name')
parser.add_argument('--binary', dest='to_binary', type=str, default='true', help='save to binary?')
parser.add_argument('--xyz-type', dest='xyz_type', type=str, default='separate', help='{\'separate\',\'joint\'}')
parser.add_argument('--xyz-name', dest='xyz_name', type=str, default='x', help='xyz_name (if xyz_joint chosen), or x name')
parser.add_argument('--y-name', dest='y_name', type=str, default='y', help='y name (if xyz_separate chosen) or unused')
parser.add_argument('--z-name', dest='z_name', type=str, default='z', help='z name (if xyz_separate chosen) or unused')
parser.add_argument('--drop-fields-if-unrecognized', dest='drop_fields', type=str, default='false', help='allow to drop a field if data type is unrecognized')
args = parser.parse_args()

file_in = args.file_in
file_out = args.file_out
to_binary = args.to_binary.lower()=="true"
xyz_types = ['separate','joint']
xyz_type = xyz_types.index(args.xyz_type)
x_name = args.xyz_name
y_name = args.y_name
z_name = args.z_name
drop_fields = args.drop_fields.lower()=="true"

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

    if (np_array.size<2):
        assert False, "No data or singe data value in field."

    
    if np_array.ndim < 2:
      vtk_array.SetNumberOfComponents(1)
      vtk_array.SetNumberOfTuples(np_array.size)
    else:
      vtk_array.SetNumberOfComponents(np_array.shape[1])
      vtk_array.SetNumberOfTuples(np_array.shape[0])

    vtk_array.SetVoidArray(np_array, np_array.size, 1)
    return vtk_array

for field in fields:
    try:
        vtk_array = np_2_vtk_array(fields[field])
        vtk_array.SetName(field)
        pointdata.AddArray(vtk_array)
    except Exception as e:
        if not drop_fields:
            raise e

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
if to_binary:
    writer.SetDataModeToBinary()
else:
    writer.SetDataModeToAscii()
writer.Update()
writer.Write()
