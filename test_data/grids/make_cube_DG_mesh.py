import math
import random
import numpy as np
from scipy.spatial import Delaunay
from netCDF4 import Dataset

VIZ = False

# geometry
height = 2.0
depth  = 2.0
width  = 2.0

# random transformations of the original mesh
random.seed(1234)
variation = .00 # as a decimal for a percent

h_all=[0.4,0.2,0.1,0.05]#,0.025,0.0125]

for key, h in enumerate(h_all):
    
    x_layer = int(width / h)+1
    y_layer = int(depth / h)+1
    z_layer = int(height / h)+1

    points = np.zeros([x_layer*y_layer*z_layer,3])

    count = 0
    for orig_x in range(x_layer):
        for orig_y in range(y_layer):
            for orig_z in range(z_layer):

                rand_pert_x = random.uniform(-variation, variation)
                rand_pert_y = random.uniform(-variation, variation)
                rand_pert_z = random.uniform(-variation, variation)
    
                x = orig_x*h
                y = orig_y*h
                z = orig_z*h
     
                if (x==0 or orig_x==x_layer-1) or (y==0 or orig_y==y_layer-1) or (z==0 or orig_z==z_layer-1):
                    points[count,:] = np.array([x,y,z])
                else:
                    points[count,:] = np.array([x+h*rand_pert_x,y+h*rand_pert_y, z+h*rand_pert_z])
                count += 1

    tet = Delaunay(points, qhull_options="Qt")

    all_cell_centers = np.zeros([tet.simplices.shape[0], 3], dtype='d')
    all_vertices_points = np.zeros([tet.simplices.shape[0], 4*3], dtype='d')
    all_adjacent_elements = np.zeros([tet.simplices.shape[0], 4], dtype='int')

    for i in range(tet.simplices.shape[0]):
         
        side_adjacency = np.empty([4], dtype='int')
        for j in range(4): 
            side_adjacency[j] = tet.neighbors[i][j]

        for j in range(4): 
            for d in range(3): 
                all_vertices_points[i,j*3+d] = points[tet.simplices[i][j], d]
            all_cell_centers[i,:] += 1./4.*points[tet.simplices[i][j], :]

        for j in range(4): 
            all_adjacent_elements[i, j] = side_adjacency[(j-1)%4] # consistent with point/edge ordering for e_type

    # write solution to netcdf
    dataset = Dataset('cube_%d.nc'%key, mode="w", clobber=True, diskless=False,\
                       persist=False, keepweakref=False, format='NETCDF4')
    dataset.createDimension('num_entities', size=tet.simplices.shape[0])
    dataset.createDimension('num_sides', size=4)
    dataset.createDimension('num_vertices', size=12)
    dataset.createDimension('spatial_dimension', size=3)
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

    dataset.createVariable('vertex_points', datatype='d', dimensions=('num_entities','num_vertices'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('adjacent_elements', datatype='int', dimensions=('num_entities','num_sides'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.createVariable('ID', datatype='int', dimensions=('num_entities'), zlib=True, complevel=8,\
                           shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                           endian='native', least_significant_digit=None, fill_value=None)

    dataset.variables['x'][:]=all_cell_centers[:,0]
    dataset.variables['y'][:]=all_cell_centers[:,1]
    dataset.variables['z'][:]=all_cell_centers[:,2]
    dataset.variables['vertex_points'][:,:]=all_vertices_points[:,:]
    dataset.variables['adjacent_elements'][:,:]=all_adjacent_elements[:,:]
    dataset.variables['ID'][:]=np.arange(tet.simplices.shape[0])

    dataset.close()

    if VIZ:

        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection


        fig = plt.figure()
        ax = fig.gca(projection='3d')

        def simplex_data(simplex, size=(1,1,1)):
            X = list()
            verts_in_simplex = list(tet.simplices[simplex])
            for face in range(0,4):
                neighbor_simplex = tet.neighbors[simplex][face]
                print(neighbor_simplex)
                if (neighbor_simplex >= 0):
                    verts_in_face = list(tet.simplices[tet.neighbors[simplex][face]])
                    intersecting_verts = list(set(verts_in_face).intersection(verts_in_simplex))
                    print(intersecting_verts)
                    print(points[intersecting_verts])
                    X.append(np.array(points[list(set(verts_in_face).intersection(verts_in_simplex))]))
                    #print(verts_in_face)
            #print(verts_in_simplex)
            X = np.array(X).astype(float)
            print(X)
            return X
        
        def plotSimplices(simplices,sizes=None,colors=None, **kwargs):
            if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(simplices)
            if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(simplices)
            g = []
            for p,s,c in zip(simplices,sizes,colors):
                g.append( simplex_data(p, size=s) )
            return Poly3DCollection(np.concatenate(g),
                                    facecolors=np.repeat(colors,6, axis=0), **kwargs)

        colors= np.random.rand(len(range(0,len(tet.simplices))),3)
        pc = plotSimplices(range(0,len(tet.simplices)),colors=colors)
        ax.add_collection3d(pc)
        ax.set_xlim3d(0,2)
        ax.set_ylim3d(0,2)
        ax.set_zlim3d(0,2)
        plt.show()

