import os
import sys
import vtk
import numpy as np
from netCDF4 import Dataset
import argparse
from vtk.util.numpy_support import numpy_to_vtk


# reads in pvtp, pvtu, or vtk meshes
# generates cell midpoints, cell vertex coordinates, and adjacent elements through sides

def convert(file_in, file_out, dim, coordinates_scale, verbose, max_verbose):
    max_verbose = (max_verbose.lower()=="true")
    verbose = (verbose.lower()=="true") or max_verbose
    if verbose:
        print("Converting file %s -> %s"%(file_in, file_out))

    dataset = Dataset(file_in, "r", format="NETCDF4")
    dimensions = dataset.dimensions
    variables = dataset.variables

    assert dim==2, "Only written for 2D currently."

    # assumed access to variables x, y, z (midpoints of cells), vertex_points (enumeration of (x,y)/(x,y,z) flattened per cell),
    # like (x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3) for nodes of a tet (3D).
    # also assumed access to variable adjacent_elements with adjacent element through an edge
    try:
        el_size=variables['x'].shape[0]
    except:
        raise Exception("Field 'x' does not exist.'")

    try:
        side_size=variables['adjacent_elements'].shape[1]
    except:
        raise Exception("Field 'adjacent_elements' does not exist.'")

    try:
        nod_size=variables['vertex_points'].shape[1]//dim
    except:
        raise Exception("Field 'vertex_coordinates' does not exist.'")


    # loop over elements, putting unique vertices in a list
    # create a field pointing at these vertex IDs rather than coordinates
    cell_vertex_ids = -np.ones(shape=(el_size, nod_size), dtype='i8') 
    count = 0
    vertex_points = variables['vertex_points']
    vertex_set = dict()
    vertex_unique_points = np.zeros(shape=(nod_size*el_size, 3), dtype='f8')
    for i in range(el_size):
        for j in range(nod_size):
            coord_pair = (float(vertex_points[i, j*dim+0]), float(vertex_points[i, j*dim+1]))
            v_id = vertex_set.get(coord_pair, -1)
            if (v_id >= 0):
                cell_vertex_ids[i,j] = v_id
            else:
                vertex_set[coord_pair] = count
                cell_vertex_ids[i,j] = count
                vertex_unique_points[count, 0] = vertex_points[i, j*dim+0]
                vertex_unique_points[count, 1] = vertex_points[i, j*dim+1]
                count += 1 
    if max_verbose:
        print(cell_vertex_ids)
    assert np.min(cell_vertex_ids)>=0, "Some vertex coordinate not found or inserted."


    # loop over elements, putting sides in a list
    adjacent_elements = variables['adjacent_elements']
    side_set = list()
    for i in range(el_size):
        for j in range(nod_size):
            adj_el = adjacent_elements[i, j]
            if (adj_el < 0):
                # unique because boundary (not shared)
                side_set.append([cell_vertex_ids[i,j], cell_vertex_ids[i,(j+1)%nod_size]])
    if max_verbose:
        print(side_set)

    writer = vtk.vtkUnstructuredGridWriter()

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(count)
    for i in range(count):
        points.InsertPoint(i, vertex_unique_points[i,0], vertex_unique_points[i,1], 0)

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)

    # insert elements
    for i in range(el_size):
        v_list = vtk.vtkIdList()
        [v_list.InsertId(j, cell_vertex_ids[i,j]) for j in range(nod_size)]
        grid.InsertNextCell(5, v_list)

    # insert sides
    for side_vert_ids in side_set:
        v_list = vtk.vtkIdList()
        [v_list.InsertId(j, side_vert_ids[j]) for j in range(len(side_vert_ids))]
        grid.InsertNextCell(3, v_list)

    # Add scalar data to the triangle
    cell_data = grid.GetCellData()
    cell_data.SetActiveScalars('Label')
    label = np.zeros(shape=(el_size+len(side_set),), dtype='i4')
    label[el_size::]=1
    vtk_label = numpy_to_vtk(label)
    vtk_label.SetName("Label")
    cell_data.SetScalars(vtk_label)
    
    grid.Modified()
    writer.SetInputData(grid)
    writer.SetFileName(file_out)
    writer.Update()
    writer.Write()
    dataset.close()

    if verbose:
        print("Complete.")

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='reads in Compadre-REBAR files and converts to unstructured grid .vtu files')
    
    parser.add_argument('--file-in', dest='file_in', type=str, help='file in name')
    parser.add_argument('--file-out', dest='file_out', type=str, help='file out name')
    parser.add_argument('--dim', dest='dim', type=int, default=2, help='spatial dimension')
    parser.add_argument('--coordinates-scale', dest='coordinates_scale', type=float, default=1.0, help='scaling of coordinates, >1 is dilation, <1 is contraction, 1 is identity')
    parser.add_argument('--verbose', dest='verbose', type=str, default='false', help='display filenames being converted')
    parser.add_argument('--max-verbose', dest='max_verbose', type=str, default='false', help='display results of conversion')
    args = parser.parse_args()

    convert(args.file_in, args.file_out, args.dim, args.coordinates_scale, args.verbose, args.max_verbose)



