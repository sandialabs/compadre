import math
import random
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import Delaunay
from scipy.integrate import quad
import quadLine1D
import quadTri2D

# helper functions
def get_velocity(coordinate):
    return np.array((np.sin(coordinate[0]), np.cos(coordinate[1])))

def get_num_points_for_order(poly_order, dimension=2):
    if (dimension==1):
        #num_points_lookup=[1,1,1,2,2]
        num_points_lookup=[item.shape[0] for item in quadLine1D.weights]#[1,1,3,4,6,7,12,13,16,19,24,27,32,36,42,55,61,66,73,78]
        num_points = num_points_lookup[poly_order]
        return num_points
    else:
        num_points_lookup=[item.shape[0] for item in quadTri2D.weights]#[1,1,3,4,6,7,12,13,16,19,24,27,32,36,42,55,61,66,73,78]
        num_points = num_points_lookup[poly_order]
        return num_points

def get_quadrature(poly_order, dimension=2):

    num_points = get_num_points_for_order(poly_order, dimension)
    w=np.empty([num_points],dtype="d")
    if (dimension==2):
        w=quadTri2D.weights[poly_order]
        q=quadTri2D.sites[poly_order]
        return (w,q)
    if (dimension==1):
        w=quadLine1D.weights[poly_order]
        q=quadLine1D.sites[poly_order]
        return (w,q)

def get_line_lengths(physical_vertices):
    pv=physical_vertices
    lengths = np.empty([3], dtype='d')
    for i in range(3):
        lengths[i] = math.sqrt(pow(pv[(i+1)%3, 0]-pv[i,0], 2) + pow(pv[(i+1)%3, 1]-pv[i,1], 2))
    return lengths

def get_triangle_area(physical_vertices):
    # return value is scaling of the physical triangle
    # relative to the reference triangle
    t1 = np.zeros([3], dtype='d')
    t2 = np.zeros([3], dtype='d')
    for i in range(2):
        #t1[i] = physical_vertices[0,i]-physical_vertices[1,i]
        #t2[i] = physical_vertices[1,i]-physical_vertices[2,i]
        t1[i] = physical_vertices[1,i]-physical_vertices[0,i]
        t2[i] = physical_vertices[2,i]-physical_vertices[0,i]
    return np.linalg.norm(np.cross(t1,t2))

def transform_reference_line_point(weights, ref_quadrature, physical_vertices, vertex_indices):
    centroid = np.zeros([2], dtype='d')
    for i in range(3):
        centroid += 1./3. * physical_vertices[i,:]

    scaling = 0.5 * get_line_lengths(physical_vertices)
    updated_weights = np.empty([3*weights.shape[0]], dtype='d')
    updated_quadrature = np.empty([3*ref_quadrature.shape[0], 2], dtype='d')
    normal_directions = np.empty([3*ref_quadrature.shape[0], 2], dtype='d')
    for i in range(3):
        for j in range(weights.shape[0]):
            updated_weights[i*weights.shape[0] + j] = scaling[i] * weights[j];
            alpha = 0.5 * (ref_quadrature[j] - (-1))
            updated_quadrature[i*ref_quadrature.shape[0] + j, :] = alpha * physical_vertices[(i+1)%3, :] + (1-alpha) * physical_vertices[i, :]
            diff_vec = physical_vertices[(i+1)%3, :] - physical_vertices[i, :]
            normal_directions[i*ref_quadrature.shape[0] + j, 0] = -diff_vec[1]
            normal_directions[i*ref_quadrature.shape[0] + j, 1] =  diff_vec[0]
            normal_directions[i*ref_quadrature.shape[0] + j, :] /= np.linalg.norm(normal_directions[i*ref_quadrature.shape[0] + j, :])

            midpoint_on_edge = 0.5 * physical_vertices[(i+1)%3, :] + 0.5 * physical_vertices[i, :]
            vector_from_center_to_edge = midpoint_on_edge - centroid
            #print( np.sign(np.dot(vector_from_center_to_edge, normal_directions[i*ref_quadrature.shape[0] + j, :])) )
            # dot product to determine if they go the same direction
            normal_directions[i*ref_quadrature.shape[0] + j, :] *= np.sign(np.dot(vector_from_center_to_edge, normal_directions[i*ref_quadrature.shape[0] + j, :]))

    return (updated_weights, updated_quadrature, normal_directions)

def transform_reference_triangle_point(weights, ref_quadrature, physical_vertices):
    scaling = get_triangle_area(physical_vertices)
    updated_weights = np.copy(weights) * scaling
    updated_quadrature = np.empty(ref_quadrature.shape, dtype='d')
    pv=physical_vertices
    a = np.array([[pv[0,0]-pv[2,0], pv[1,0]-pv[2,0]], [pv[0,1]-pv[2,1], pv[1,1]-pv[2,1]]], dtype='f8');
    #ainv = np.linalg.inv(np.matrix(a))
    #ainv = np.matrix(a)
    for i in range(ref_quadrature.shape[0]):
        updated_quadrature[i,:] = np.matmul(a, ref_quadrature[i,:]) + pv[2,:].T
        #updated_quadrature[i,:] = np.dot(a, ref_quadrature[i,:]) + pv[2,:].T
    return (updated_weights, updated_quadrature)
    
def get_unit_normal_vector(line_coordinates):
    # (x0,y0,x1,y1) are given

    # first get the normal (reused for all quadrature)
    # tangent vector is [x1-x0,y1-y0] so normal is [(y1-y0),-(x1-x0)]
    unit_normal_vector = np.array((line_coordinates[3]-line_coordinates[1], line_coordinates[0]-line_coordinates[2]))
    unit_normal_vector = unit_normal_vector / np.linalg.norm(unit_normal_vector, ord=2)

    # then the tangent
    unit_tangent_vector = np.array((line_coordinates[2]-line_coordinates[0], line_coordinates[3]-line_coordinates[1]))
    unit_tangent_vector = unit_tangent_vector / np.linalg.norm(unit_tangent_vector, ord=2)

    # is the normal pointing correct direction? 
    unit_normal_vector = np.concatenate((unit_normal_vector, np.zeros([1])), axis=0) 
    unit_tangent_vector = np.concatenate((unit_tangent_vector, np.zeros([1])), axis=0) 
    
    # take cross product of normal as calculated and tangent and check the sign in the z component
    cross_product = np.cross(unit_normal_vector, unit_tangent_vector)
    if (cross_product[2] > 0):
        unit_normal_vector = unit_normal_vector[0:2]
    else:
        unit_normal_vector = -unit_normal_vector[0:2]
    return unit_normal_vector

def get_boundary_type(edge_adjacency):
    # 0 is interior edge
    # 2 is exterior edge
    output = np.empty([3], dtype='int')
    for key, val in enumerate(edge_adjacency):
        if val>=0:
            # consistent with point->edge iteration
            output[(key+1)%3] = 0
        else:
            output[(key+1)%3] = 2
    return output

def integrand(x, line_coordinates, unit_normal_vector):
    # convert the x which is a quadrature for the interval
    # need transformation from [0,1] to the line segment from (x0,y0) to (x1,y1)

    # calculate quadrature_coordinate
    quadrature_coordinate = x*line_coordinates[0:2] + (1-x)*line_coordinates[2:4]
    
    # diagnostics
    # first get the quadrature
    # then the tangent
    #unit_tangent_vector = np.array((line_coordinates[2]-line_coordinates[0], line_coordinates[3]-line_coordinates[1]))
    #unit_tangent_vector = unit_tangent_vector / np.linalg.norm(unit_tangent_vector, ord=2)
    #print np.dot(unit_normal_vector, unit_tangent_vector)

    velocity = get_velocity(quadrature_coordinate)
 
    # line is stretch going from quadrature on [0,1] to the 2d line, but it is linear
    domain_stretch = np.linalg.norm(line_coordinates[2:4]-line_coordinates[0:2], ord=2) # accounts for the fact that our x is undergoing a coordinate transformation

    return np.dot(unit_normal_vector, velocity)*domain_stretch

def integrate_along_line(line_coordinates):
    # (x0,y0,x1,y1) are given

    unit_normal_vector = get_unit_normal_vector(line_coordinates)

    # return the integral
    return quad(integrand, 0, 1, args=(line_coordinates, unit_normal_vector))[0]

vis = False

# geometry
height = 2.0
width  = 2.0

# random transformations of the original mesh
random.seed(1234)
blowup_ratio = 1 # 1 does nothing, identity
random_rotation = False
rotation_max = 180 # in degrees (either clockwise or counterclockwise, 180 should be highest needed)
variation = .00 # as a decimal for a percent


#h_all=[0.2]#,0.1,0.05,0.025,0.0125,0.00625]
h_all=[0.4,0.2,0.1,0.05,0.025,0.0125]

poly_order = 4
num_points_interior = get_num_points_for_order(poly_order, 2)
num_points_exterior = get_num_points_for_order(poly_order, 1)

for key, h in enumerate(h_all):
    
    x_layer = int(width / h)+1
    y_layer = int(height / h)+1

    flagged_layers_left_and_right = .02*x_layer
    flagged_layers_top_and_bottom = .02*y_layer
    
    points = np.zeros([x_layer*y_layer,2])

    for vertical in range(y_layer):
        for horizontal in range(x_layer):
            rand_pert_x = random.uniform(-variation, variation)
            rand_pert_y = random.uniform(-variation, variation)
    
            x = horizontal*h
            y = vertical*h
     
            if (horizontal==0 or horizontal==x_layer-1) or (vertical==0 or vertical==y_layer-1):
                points[x_layer*vertical + horizontal,:] = np.array([x,y])
            else:
                points[x_layer*vertical + horizontal,:] = np.array([x+h*rand_pert_x,y+h*rand_pert_y])

    tri = Delaunay(points, qhull_options="Qt")


    
    all_weights = np.empty([tri.simplices.shape[0], num_points_interior + 3*num_points_exterior], dtype='d')
    all_quadrature = np.empty([tri.simplices.shape[0], 2*(num_points_interior + 3*num_points_exterior)], dtype='d')
    all_normals = np.zeros([tri.simplices.shape[0], 2*(num_points_interior + 3*num_points_exterior)], dtype='d')
    all_interior = np.ones([tri.simplices.shape[0], num_points_interior + 3*num_points_exterior], dtype='int')
    all_vertices = np.zeros([tri.simplices.shape[0], 2], dtype='d')
    all_vertices_points = np.zeros([tri.simplices.shape[0], 6], dtype='d')
    all_adjacent_elements = np.zeros([tri.simplices.shape[0], 3], dtype='int')

    (w_interior, q_interior) = get_quadrature(poly_order, 2)
    (w_exterior, q_exterior) = get_quadrature(poly_order, 1)
    for i in range(tri.simplices.shape[0]): # all triangles
        physical_vertices = np.empty([3,2], dtype='d')
        vertex_indices = np.empty([3], dtype='int')
        edge_adjacency = np.empty([3], dtype='int')
        for j in range(3): # vertices
            physical_vertices[j,:] = points[tri.simplices[i][j], :]
            vertex_indices[j] = tri.simplices[i][j]
            all_vertices[i,:] += 1./3.*points[tri.simplices[i][j], :]
            edge_adjacency[j] = tri.neighbors[i][j]
            all_vertices_points[i,j*2+0] = points[tri.simplices[i][j], 0]
            all_vertices_points[i,j*2+1] = points[tri.simplices[i][j], 1]

        (w_physical_interior, q_physical_interior) = transform_reference_triangle_point(w_interior, q_interior, physical_vertices) 
        for j in range(num_points_interior):
            all_quadrature[i, 2*j  ] = q_physical_interior[j, 0]
            all_quadrature[i, 2*j+1] = q_physical_interior[j, 1]
            all_weights   [i, j    ] = w_physical_interior[j]

        (w_physical_exterior, q_physical_exterior, q_normals_exterior) = transform_reference_line_point(w_exterior, q_exterior, physical_vertices, vertex_indices) 
        e_type = get_boundary_type(edge_adjacency)
        for k in range(3): # edges
            for j in range(num_points_exterior):
                all_quadrature[i, 2*num_points_interior + 2*k*num_points_exterior + 2*j  ] = q_physical_exterior[k*num_points_exterior + j, 0]
                all_quadrature[i, 2*num_points_interior + 2*k*num_points_exterior + 2*j+1] = q_physical_exterior[k*num_points_exterior + j, 1]
                all_normals   [i, 2*num_points_interior + 2*k*num_points_exterior + 2*j  ] = q_normals_exterior[k*num_points_exterior + j, 0]
                all_normals   [i, 2*num_points_interior + 2*k*num_points_exterior + 2*j+1] = q_normals_exterior[k*num_points_exterior + j, 1]
                # put a 1 for interior quadrature point to a cell, 0 for exterior (edge/face), and 2 for exterior (edge/face) but on a Dirichlet boundary
                all_interior  [i, num_points_interior + k*num_points_exterior + j] = e_type[k];
                all_weights   [i, num_points_interior + k*num_points_exterior + j] = w_physical_exterior[k*num_points_exterior + j]
            all_adjacent_elements [i, k] = edge_adjacency[(k-1)%3] # consistent with point/edge ordering for e_type


    if (vis):
        #for z in range(all_quadrature.shape[0]):
        # visualization
        import matplotlib.pyplot as plt
        from matplotlib import collections as mc
        #plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
        all_lines = list()
        fig, ax = plt.subplots()
        ax.plot(points[:,0], points[:,1], 'o')
        for i in range(num_points_interior):
            ax.plot(all_quadrature[:,2*i + 0], all_quadrature[:,2*i + 1], 'o')
        for k in range(3): # edges
            for j in range(num_points_exterior):
                ax.plot(all_quadrature[:,2*(num_points_interior + k*num_points_exterior + j) + 0], all_quadrature[:,2*(num_points_interior + k*num_points_exterior + j) + 1], 'o')
        #for i in range(z,z+1):#all_quadrature.shape[0]):
        for i in range(all_quadrature.shape[0]):
            for k in range(3):
                for j in range(num_points_exterior):
                    all_lines.append([[all_quadrature[i,2*(num_points_interior + k*num_points_exterior + j) + 0], all_quadrature[i,2*(num_points_interior + k*num_points_exterior + j) + 1]], [0.1*h*all_normals[i,2*(num_points_interior + k*num_points_exterior + j) + 0]+all_quadrature[i,2*(num_points_interior + k*num_points_exterior + j) + 0], 0.1*h*all_normals[i,2*(num_points_interior + k*num_points_exterior + j) + 1]+all_quadrature[i,2*(num_points_interior + k*num_points_exterior + j) + 1]]])

        lc = mc.LineCollection(all_lines, linewidths=2)
        ax.add_collection(lc)
        ax.triplot(points[:,0], points[:,1], tri.simplices.copy())
        ax.autoscale()
        ax.margins(0.1)
        ax.axis('equal')
        plt.show()


    # write solution to netcdf
    dataset = Dataset('dg_%d.nc'%key, mode="w", clobber=True, diskless=False,\
                       persist=False, keepweakref=False, format='NETCDF4')
    dataset.createDimension('num_entities', size=tri.simplices.shape[0])
    dataset.createDimension('num_edges', size=3)
    dataset.createDimension('num_vertices', size=6)
    #dataset.createDimension('num_interior_quadrature', size=num_points_interior)
    #dataset.createDimension('num_exterior_quadrature', size=3*num_points_exterior)
    #dataset.createDimension('num_total_quadrature', size=num_points_interior + 3*num_points_exterior)
    #dataset.createDimension('vector_for_quadrature', size=2*(num_points_interior + 3*num_points_exterior)) # a vector at each quadrature point
    #dataset.createDimension('scalar_for_quadrature', size=num_points_interior + 3*num_points_exterior) # a scalar at each quadrature point
    dataset.createDimension('spatial_dimension', size=2) # 2 is spatial dimension
    dataset.createDimension('scalar_dim', size=1) 

    dataset.createVariable('x', datatype='d', dimensions=('num_entities'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('y', datatype='d', dimensions=('num_entities'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('z', datatype='d', dimensions=('num_entities'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    #dataset.createVariable('quadrature_weights', datatype='d', dimensions=('num_entities','scalar_for_quadrature'), zlib=True, complevel=8,\
    #                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
    #                       endian='native', least_significant_digit=None, fill_value=None)

    #dataset.createVariable('quadrature_points', datatype='d', dimensions=('num_entities','vector_for_quadrature'), zlib=True, complevel=8,\
    #                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
    #                       endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('vertex_points', datatype='d', dimensions=('num_entities','num_vertices'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    #dataset.createVariable('unit_normal', datatype='d', dimensions=('num_entities','vector_for_quadrature'), zlib=True, complevel=8,\
    #                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
    #                       endian='native', least_significant_digit=None, fill_value=None)

    #dataset.createVariable('interior', datatype='int', dimensions=('num_entities','scalar_for_quadrature'), zlib=True, complevel=8,\
    #                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
    #                       endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('adjacent_elements', datatype='int', dimensions=('num_entities','num_edges'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('ID', datatype='int', dimensions=('num_entities'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.variables['x'][:]=all_vertices[:,0]
    dataset.variables['y'][:]=all_vertices[:,1]
    dataset.variables['z'][:]=np.zeros(all_vertices[:,1].shape)
    #dataset.variables['quadrature_weights'][:,:]=all_weights[:,:]
    #dataset.variables['quadrature_points'][:,:]=all_quadrature[:,:]
    dataset.variables['vertex_points'][:,:]=all_vertices_points[:,:]
    #dataset.variables['unit_normal'][:,:]=all_normals[:,:]
    #dataset.variables['interior'][:,:]=all_interior[:,:]
    dataset.variables['adjacent_elements'][:,:]=all_adjacent_elements[:,:]
    dataset.variables['ID'][:]=np.arange(tri.simplices.shape[0])


    #help(dataset)
    dataset.close()

#print (get_quadrature(3))


