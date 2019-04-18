import math
import random
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import Delaunay
from scipy.integrate import quad

# helper functions

def get_scalar(coordinate):
    return np.sin(coordinate[0])*np.sin(coordinate[1])

def get_velocity(coordinate):
    #return np.array((np.sin(coordinate[0]), np.cos(coordinate[1])))
    return np.array((get_scalar(coordinate), -get_scalar(coordinate)))

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

vis = True

# geometry
height = 1.0
width  = 1.0

# random transformations of the original mesh
random.seed(1234)
blowup_ratio = 1 # 1 does nothing, identity
random_rotation = False
rotation_max = 180 # in degrees (either clockwise or counterclockwise, 180 should be highest needed)
variation = .00 # as a decimal for a percent


h_all=[0.2]#,0.1,0.05,0.025,0.0125,0.00625]


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

    # if (vis):
    #     visualization
    #     import matplotlib.pyplot as plt
    #     plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    #     plt.plot(points[:,0], points[:,1], 'o')
    #     plt.show()

    # calculate all lines
    all_lines = set()
    for i in range(tri.simplices.shape[0]): # all triangles
        for j in range(3): # vertices
            pair = [tri.simplices[i][j], tri.simplices[i][(j+1) % 3]]
            pair.sort()
            pair = tuple(pair)
            all_lines.add(pair)
    lines = np.ndarray([len(all_lines),2], dtype='int32')
    for i, line in enumerate(all_lines):
        lines[i,:] = np.array(line)

    #if (vis):
    #    # visualization of directed edges
    #    import matplotlib.pyplot as plt
    #    from matplotlib import collections as mc
    #    from matplotlib.pyplot import arrow as arrow
    #    fig, ax = plt.subplots()
    #    line_list = []
    #    for row in lines:
    #        line_list.append([[points[row[0],0], points[row[0],1]],[points[row[1],0], points[row[1],1]]])
    #        #origin = line_list[-1][0]
    #        #diff = list(points[row[1],:]-points[row[0],:])
    #        #ax.arrow(origin[0], origin[1], diff[0], diff[1])
    #    lc = mc.LineCollection(line_list, linewidths=2)
    #    #X = points[lines[:,0],0]
    #    #Y = points[lines[:,0],1]
    #    #print X, Y
    #    #U = points[lines[:,1],0] - points[lines[:,0],0]
    #    ##U = np.zeros(lines.shape[0]) #points[lines[:,1],0] - points[lines[:,0],0]
    #    #V = points[lines[:,1],1] - points[lines[:,0],1]
    #    ##V = np.ones(lines.shape[0]) #points[lines[:,1],1] - points[lines[:,0],1]
    #    #q = ax.quiver(X, Y, U, V, scale=None)
    #    ax.add_collection(lc)
    #    ax.autoscale()
    #    ax.margins(0.1)
    #    plt.show()

    #get new points for the lines by copying from coordinates
    #freeing the edges from being attached to one another
    #now transform by their midpoint
    new_line_points = np.concatenate((points[lines[:,0],:],points[lines[:,1],:]),axis=1)
    unit_normal_vectors = np.zeros([new_line_points.shape[0],lines.shape[1]])
    solution_on_line = np.zeros(new_line_points.shape[0])
    for i, row in enumerate(new_line_points):

        midpoint = 0.5*row[0:2] + 0.5*row[2:4]
        scaled_midpoint = blowup_ratio*midpoint

        # remove old midpoint
        new_line_points[i,0:2] -= midpoint
        new_line_points[i,2:4] -= midpoint

        # generate random rotation matrix
        if (random_rotation):
            theta = rotation_max*math.pi/180.0*(random.random()-.5)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c,-s), (s, c)))
            t_vec = new_line_points[i,0:2]
            new_line_points[i,0:2] = np.dot(np.transpose(R), t_vec)
            t_vec = new_line_points[i,2:4]
            new_line_points[i,2:4] = np.dot(np.transpose(R), t_vec)

        # insert scaled midpoint
        new_line_points[i,0:2] += scaled_midpoint
        new_line_points[i,2:4] += scaled_midpoint
        
        # integrate v against normal along this line
        solution_on_line[i] = integrate_along_line(new_line_points[i,:])
        unit_normal_vectors[i] = get_unit_normal_vector(new_line_points[i,:])

    if (vis):
        # visualization of directed edges
        import matplotlib.pyplot as plt
        from matplotlib import collections as mc
        from matplotlib.pyplot import arrow as arrow
        fig, ax = plt.subplots()
        line_list = []
        for row in new_line_points:
            #line_list.append([[points[row[0],0], points[row[0],1]],[points[row[1],0], points[row[1],1]]])
            line_list.append([row[0:2],row[2:4]]) #row[[points[row[0],0], points[row[0],1]],[points[row[1],0], points[row[1],1]]])
            #origin = line_list[-1][0]
            #diff = list(points[row[1],:]-points[row[0],:])
            #ax.arrow(origin[0], origin[1], diff[0], diff[1])
        lc = mc.LineCollection(line_list, linewidths=2)
        #X = points[lines[:,0],0]
        #Y = points[lines[:,0],1]
        #print X, Y
        #U = points[lines[:,1],0] - points[lines[:,0],0]
        ##U = np.zeros(lines.shape[0]) #points[lines[:,1],0] - points[lines[:,0],0]
        #V = points[lines[:,1],1] - points[lines[:,0],1]
        ##V = np.ones(lines.shape[0]) #points[lines[:,1],1] - points[lines[:,0],1]
        #q = ax.quiver(X, Y, U, V, scale=None)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        plt.show()

    # write solution to netcdf
    dataset = Dataset('rt0_%d.nc'%key, mode="w", clobber=True, diskless=False,\
                       persist=False, keepweakref=False, format='NETCDF4')
    dataset.createDimension('num_entities', size=new_line_points.shape[0])
    dataset.createDimension('related_coordinates_size', size=2*2) # 2 is spatial description and a 1d dimension entity has two endpoint to describe
    dataset.createDimension('entity_dimension', size=1) # 1 is entity dimension
    dataset.createDimension('spatial_dimension', size=2) # 2 is spatial dimension
    dataset.setncattr('entity_dimension',int(1)) # lines are 1d objects
    dataset.setncattr('spatial_dimension',int(2)) # represented in 2D space

    dataset.createVariable('related_coordinates', datatype='d', dimensions=('num_entities','related_coordinates_size'), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('unit_normals', datatype='d', dimensions=('num_entities','spatial_dimension'), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('x', datatype='d', dimensions=('num_entities'), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('y', datatype='d', dimensions=('num_entities'), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('z', datatype='d', dimensions=('num_entities'), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('u', datatype='d', dimensions=('num_entities'), zlib=False, complevel=4,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.variables['related_coordinates'][:,:]=new_line_points[:,:]
    dataset.variables['unit_normals'][:,:]=unit_normal_vectors[:,:]
    dataset.variables['x'][:]=0.5*new_line_points[:,0:1] + 0.5*new_line_points[:,2:3]
    dataset.variables['y'][:]=0.5*new_line_points[:,1:2] + 0.5*new_line_points[:,3:4]
    dataset.variables['z'][:]=0.0*new_line_points[:,0:1]
    dataset.variables['u'][:]=solution_on_line[:]

    #help(dataset)
    dataset.close()


