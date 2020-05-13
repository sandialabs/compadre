import os
import sys
import vtk
import numpy as np
from netCDF4 import Dataset
import argparse
from numba import jit

@jit(nopython=True,parallel=False)
def getFlattenCellVertices(i, new_data, coords, nodes, dim):
    nod_size = len(nodes)
    for j in range(nod_size):
        for k in range(dim):
            new_data[dim*j+k] = coords[nodes[j],k]

@jit(nopython=True,parallel=False)
def getNewMidpoints(i, new_data, coords, nodes):
    nod_size = len(nodes)
    for j in range(nod_size):
        for k in range(3):
            new_data[k] += coords[nodes[j], k] / float(nod_size)

@jit(nopython=True,parallel=False)
def getAdjacentCell(i, vert_indices, connect, el_size, nod_size, dim):
    # loop all other elements
    for k in range(i+1,el_size):
        # grab two nodes at a time
        for l in range(nod_size):
            k_verts = [connect[k,l], connect[k,(l+1)%nod_size]]
            k_verts.sort()
            same = True
            for m in range(dim):
                if vert_indices[m]!=k_verts[m]:
                    same = False
            if same:
                return (k, l)
    return (-1, -1)

# reads in pvtp, pvtu, or vtk meshes
# generates cell midpoints, cell vertex coordinates, and adjacent elements through sides

def convert(file_in, file_out, dim, coordinates_scale, verbose, max_verbose):
    max_verbose = (max_verbose.lower()=="true")
    verbose = (verbose.lower()=="true") or max_verbose
    if verbose:
        print("Converting file %s -> %s"%(file_in, file_out))

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
        reader = vtk.vtkUnstructuredGridReader()
    elif (file_extension==".vtk"):
        reader = vtk.vtkUnstructuredGridReader()
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
    print(coords)

    # scaling of coordinates
    if (coordinates_scale!=1.0):
        coords = coords * coordinates_scale
    
    el_size = 0
    n_any_kind_of_cell = data.GetNumberOfCells()
    for i in range(n_any_kind_of_cell):
        if data.GetCell(i).GetCellType()==5 and dim==2: # triangle
            el_size += 1
        if data.GetCell(i).GetCellType()==7 and dim==3: # tet
            el_size += 1

    cell_list = [None,]*el_size
    nod_size = 3 if dim==2 else 4
    connect = np.zeros(shape=(el_size,nod_size), dtype='i8')
    count = 0
    for i in range(n_any_kind_of_cell):
        if data.GetCell(i).GetCellType()==5 and dim==2: # triangle
            cell_list[count] = i
            connect[count,:] = np.array([data.GetCell(cell_list[i]).GetPointIds().GetId(n) for n in range(nod_size)])
            count += 1
        if data.GetCell(i).GetCellType()==7 and dim==3: # tet
            cell_list[count] = i
            connect[count,:] = np.array([data.GetCell(cell_list[i]).GetPointIds().GetId(n) for n in range(nod_size)])
            count += 1

    # get cell centers from coordinates
    midpoints = np.zeros(shape=(el_size,3), dtype='f8')
    for i in range(el_size):
        this_new_midpoint_coords = np.zeros(3)
        getNewMidpoints(i, this_new_midpoint_coords, coords, connect[i,:])
        midpoints[i,:] = this_new_midpoint_coords
    if max_verbose:
        print(midpoints)

    vertex_coordinates = np.zeros(shape=(el_size, dim*nod_size),dtype='f8')
    for i in range(el_size):
        this_extra_data = np.zeros(dim*nod_size)
        getFlattenCellVertices(i, this_extra_data, coords, connect[i,:], dim)
        vertex_coordinates[i,:] = this_extra_data
    if max_verbose:
        print(vertex_coordinates)

    if (dim==2):
        # only works for simplices
        adjacent_elements = -np.ones(shape=(el_size, dim+1),dtype='i8') 
        # loop over all elements
        i_verts = np.ones(shape=(2,), dtype='i4')
        for i in range(el_size):
            # grab two nodes at a time
            for j in range(nod_size):
                if (adjacent_elements[i, j]==-1):
                    i_verts[0] = min(connect[i,j], connect[i,(j+1)%nod_size])
                    i_verts[1] = max(connect[i,j], connect[i,(j+1)%nod_size])
                    (element, edge) = getAdjacentCell(i, i_verts, connect, el_size, nod_size, dim)
                    if (element>-1):
                        adjacent_elements[i, j] = element
                        adjacent_elements[element, edge] = i
        if max_verbose:
            print(adjacent_elements)
    
    # all point data now collected (cell centers + vertices oordinates + adjacencies to cells through sides)
    # write solution to netcdf
    dataset = Dataset(file_out, mode="w", clobber=True, diskless=False,\
                       persist=False, keepweakref=False, format='NETCDF4')
    
    dataset.createDimension('num_entities', size=el_size)
    dataset.createDimension('num_sides', size=dim+1)
    dataset.createDimension('num_vertex_coords', size=nod_size*dim)
    dataset.createDimension('spatial_dimension', size=dim)
    dataset.createDimension('scalar_dim', size=1) 
    
    dataset.createVariable('x', datatype='d', dimensions=('num_entities'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)
    
    if (dim>1):
        dataset.createVariable('y', datatype='d', dimensions=('num_entities'), zlib=True, complevel=8,\
                               shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                               endian='native', least_significant_digit=None, fill_value=None)
    
    if (dim>2):
        dataset.createVariable('z', datatype='d', dimensions=('num_entities'), zlib=True, complevel=8,\
                               shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                               endian='native', least_significant_digit=None, fill_value=None)
    
    dataset.createVariable('vertex_points', datatype='d', dimensions=('num_entities','num_vertex_coords'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)
    
    if (dim==2):
        dataset.createVariable('adjacent_elements', datatype='int', dimensions=('num_entities','num_sides'), zlib=True, complevel=8,\
                               shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                               endian='native', least_significant_digit=None, fill_value=None)
    
    dataset.createVariable('ID', datatype='int', dimensions=('num_entities'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)
    
    dataset.variables['x'][:]=midpoints[:,0]
    if (dim>1):
        dataset.variables['y'][:]=midpoints[:,1]
    if (dim>2):
        dataset.variables['z'][:]=midpoints[:,2]
    
    dataset.variables['vertex_points'][:,:]=vertex_coordinates[:,:]
    if (dim==2):
        dataset.variables['adjacent_elements'][:,:]=adjacent_elements[:,:]
    dataset.variables['ID'][:]=np.arange(el_size)
    
    dataset.close()
    if verbose:
        print("Complete.")

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='reads in VTK files and converts them to Compadre-REBAR style netCDF files')
    
    parser.add_argument('--file-in', dest='file_in', type=str, help='file in name')
    parser.add_argument('--file-out', dest='file_out', type=str, help='file out name')
    parser.add_argument('--dim', dest='dim', type=int, default=3, help='spatial dimension')
    parser.add_argument('--coordinates-scale', dest='coordinates_scale', type=float, default=1.0, help='scaling of coordinates, >1 is dilation, <1 is contraction, 1 is identity')
    parser.add_argument('--verbose', dest='verbose', type=str, default='false', help='display filenames being converted')
    parser.add_argument('--max-verbose', dest='max_verbose', type=str, default='false', help='display results of conversion')
    args = parser.parse_args()

    convert(args.file_in, args.file_out, args.dim, args.coordinates_scale, args.verbose, args.max_verbose)



