import os
import sys
import vtk

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
elif (file_extension==".vtu"):
    reader = vtk.vtkXMLUnstructuredGridReader()
else:
    assert False, "file input extension name not recognized"

reader.SetFileName(file_in)
reader.Update()

data= reader.GetOutput()
pd = data.GetPointData()
help(pd)

#fields = dict()
#n_xyz = pd.GetNumberOfTuples()
#for i in range(n_xyz):
#    data = 0
#    pd.GetTuple(i,data)
#    print(data)

#d_xyz = pd.GetNumberOfComponents()
print(n_xyz)#,d_xyz)
#np = pd.GetNumberOfPoints()
#GetNumberOfTuples
#GetNumberOfComponents
for i in range(pd.GetNumberOfArrays()):
    fields[pd.GetArrayName(i)]=1
    print(pd.GetArrayName(i))
print(fields)
