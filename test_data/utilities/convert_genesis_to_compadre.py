import os
import sys
import numpy as np
from netCDF4 import Dataset
import argparse
from numba import jit

@jit(nopython=True,parallel=False)
def getFlattenCellVertices(i, new_data, coords, connect, nod_size, dim):
    for j in range(nod_size):
        for k in range(dim):
            new_data[dim*j+k] = coords[connect[i][j]-1,k]

@jit(nopython=True,parallel=False)
def getNewMidpoints(i, new_data, coords, connect, nod_size, dim):
    for j in range(nod_size):
        for k in range(dim):
            new_data[k] += coords[connect[i,j]-1, k] / float(nod_size)

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

def convert(file_in, file_out, xyz_type_str, x_name, y_name, z_name, cell_to_vertex_field_name, dimension_nodes_per_element_name, dim, coordinates_scale, verbose, max_verbose, num_blocks = 1, cell_to_vertex_field_name2="", dimension_nodes_per_element_name2=""):
    if cell_to_vertex_field_name2=="": 
        cell_to_vertex_field_name2 = cell_to_vertex_field_name
    if dimension_nodes_per_element_name2=="": 
        dimension_nodes_per_element_name2 = dimension_nodes_per_element_name

    xyz_types = ['separate','joint']
    xyz_type = xyz_types.index(xyz_type_str)
    max_verbose = (max_verbose.lower()=="true")
    verbose = (verbose.lower()=="true") or max_verbose

    dataset = Dataset(file_in, "r", format="NETCDF4")
    dimensions = dataset.dimensions
    variables = dataset.variables
    if verbose:
        print("Converting file %s -> %s"%(file_in, file_out))
    
    # get dimensions of coordinates
    n_xyz = 0
    for variable_name in variables:
        if (str(variable_name)==x_name):
            assert len(variables[variable_name].shape)<=2, "Doesn't support fields with more than two axis."
            n_xyz = variables[variable_name].shape[0]
    
    # copy coordinates
    coords = np.zeros(shape=(n_xyz, 3), dtype='f8')
    coord_x_found = False
    coord_y_found = False
    coord_z_found = False
    for variable_name in variables:
        if (str(variable_name)==x_name):
            if (xyz_type==0):
                coords[:,0] = variables[variable_name][:]
                coord_x_found = True
            else:
                coords[:,:] = variables[variable_name][:,:]
                coord_x_found = True
                coord_y_found = True
                coord_z_found = True
        if (str(variable_name)==y_name):
            if (xyz_type==0):
                coords[:,1] = variables[variable_name][:]
                coord_y_found = True
        if (str(variable_name)==z_name):
            if (xyz_type==0):
                coords[:,2] = variables[variable_name][:]
                coord_z_found = True
    
    assert coord_x_found, "Field for --xyz-name, %s, was not found."%x_name
    if (dim>1):
        assert coord_y_found, "Field for --y-name, %s, was not found."%y_name
    if (dim>2):
        assert coord_z_found, "Field for --z-name, %s, was not found."%z_name
    
    # scaling of coordinates
    if (coordinates_scale!=1.0):
        coords = coords * coordinates_scale

    list_of_midpoints_blocks = [None,]*num_blocks
    list_of_vertex_coordinates_blocks = [None,]*num_blocks
    list_of_adjacent_elements_blocks = [None,]*num_blocks
    for block_num in range(num_blocks):
        if block_num==0:
            tmp_cell_to_vertex_field_name = cell_to_vertex_field_name
            tmp_dimension_nodes_per_element_name = dimension_nodes_per_element_name
        if block_num==1:
            tmp_cell_to_vertex_field_name = cell_to_vertex_field_name2
            tmp_dimension_nodes_per_element_name = dimension_nodes_per_element_name2
    
        #print(coords)
        connect = np.array(variables[tmp_cell_to_vertex_field_name])
        el_size = connect.shape[0]
        nod_size = int(dimensions[tmp_dimension_nodes_per_element_name].size)
        
        # get cell centers from coordinates
        midpoints = np.zeros(shape=(el_size,3), dtype='f8')
        for i in range(el_size):
            this_new_midpoint_coords = np.zeros(3)
            getNewMidpoints(i, this_new_midpoint_coords, coords, connect, nod_size, dim)
            midpoints[i,:] = this_new_midpoint_coords
        if max_verbose:
            print(midpoints)
        list_of_midpoints_blocks[block_num] = midpoints
        
        vertex_coordinates = np.zeros(shape=(el_size, dim*nod_size),dtype='f8')
        for i in range(el_size):
            this_extra_data = np.zeros(dim*nod_size)
            getFlattenCellVertices(i, this_extra_data, coords, connect, nod_size, dim)
            vertex_coordinates[i,:] = this_extra_data
        if max_verbose:
            print(vertex_coordinates)
        list_of_vertex_coordinates_blocks[block_num] = vertex_coordinates
        
        if (dim==2):
            # only works for simplices
            adjacent_elements = -np.ones(shape=(el_size, dim+1),dtype='i8') 
            # loop over all elements
            i_verts = np.ones(shape=(2,), dtype='i4')
            for i in range(el_size):
                # grab two nodes at a time
                for j in range(nod_size):
                    #i_verts = [connect[i,j], connect[i,(j+1)%nod_size]]
                    #i_verts.sort()
                    if (adjacent_elements[i, j]==-1):
                        i_verts[0] = min(connect[i,j], connect[i,(j+1)%nod_size])
                        i_verts[1] = max(connect[i,j], connect[i,(j+1)%nod_size])
                        (element, edge) = getAdjacentCell(i, i_verts, connect, el_size, nod_size, dim)
                        if (element>-1):
                            adjacent_elements[i, j] = element
                            adjacent_elements[element, edge] = i
            if max_verbose:
                print(adjacent_elements)
            list_of_adjacent_elements_blocks[block_num] = adjacent_elements

    # merge all blocks info
    midpoints = np.vstack(list_of_midpoints_blocks)
    vertex_coordinates = np.vstack(list_of_vertex_coordinates_blocks)
    adjacent_elements = np.vstack(list_of_adjacent_elements_blocks)
    el_size = midpoints.shape[0]
    
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

    parser = argparse.ArgumentParser(description='reads in genesis files\
        default for coordinates is that they are stored in field variables \'x\', \'y\', and \'z\'\
        (can be modified with command line argumentstake diff of field on file1 and file2 and store in new-file')
    
    parser.add_argument('--file-in', dest='file_in', type=str, help='file in name')
    parser.add_argument('--file-out', dest='file_out', type=str, help='file out name')
    parser.add_argument('--xyz-type', dest='xyz_type', type=str, default='separate', help='{\'separate\',\'joint\'}')
    parser.add_argument('--xyz-name', dest='xyz_name', type=str, default='x', help='xyz_name (if xyz_joint chosen), or x name')
    parser.add_argument('--y-name', dest='y_name', type=str, default='y', help='y name (if xyz_separate chosen) or unused')
    parser.add_argument('--z-name', dest='z_name', type=str, default='z', help='z name (if xyz_separate chosen) or unused')
    parser.add_argument('--cell-to-vertex-field-name', dest='cell_to_vertex_field_name', type=str, default='connect1', help='name for dimension of cells')
    parser.add_argument('--dimension-nodes-per-element-name', dest='dimension_nodes_per_element_name', type=str, default='num_nod_per_el1', help='name for dimension of nodes per element')
    parser.add_argument('--cell-to-vertex-field-name2', dest='cell_to_vertex_field_name2', type=str, default='connect1', help='name for dimension of cells')
    parser.add_argument('--dimension-nodes-per-element-name2', dest='dimension_nodes_per_element_name2', type=str, default='num_nod_per_el1', help='name for dimension of nodes per element')
    parser.add_argument('--dim', dest='dim', type=int, default=3, help='spatial dimension')
    parser.add_argument('--num-blocks', dest='num_blocks', type=int, default=1, help='num blocks (default 1)')
    parser.add_argument('--coordinates-scale', dest='coordinates_scale', type=float, default=1.0, help='scaling of coordinates, >1 is dilation, <1 is contraction, 1 is identity')
    parser.add_argument('--verbose', dest='verbose', type=str, default='false', help='display filenames being converted')
    parser.add_argument('--max-verbose', dest='max_verbose', type=str, default='false', help='display results of conversion')
    args = parser.parse_args()

    convert(args.file_in, args.file_out, args.xyz_type, args.xyz_name, args.y_name, args.z_name, args.cell_to_vertex_field_name, args.dimension_nodes_per_element_name, args.dim, args.coordinates_scale, args.verbose, args.max_verbose, args.num_blocks, args.cell_to_vertex_field_name2, args.dimension_nodes_per_element_name2)
