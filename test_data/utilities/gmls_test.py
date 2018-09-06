import sys
import numpy as np
import math
#import matplotlib.pyplot as plt

from GMLS import gmls

def test1():
    # 3rd order reconstruction using 2nd order basis in 3D
    gmls_obj=gmls.GMLS_Python(2, "QR", 2, 3)
    gmls_obj.setWeightingOrder(int(8))
    
    
    neighbor_lists = np.array([[1000,] + range(1000)], dtype=np.dtype(int))
    target_sites = np.array([[0,0,0],], dtype=np.dtype('d'))
    
    
    ND = 10
    nx, ny, nz = (ND, ND, ND)
    dx = np.linspace(-1, 1, nx)
    dy = np.linspace(-1, 1, ny)
    dz = np.linspace(-1, 1, nz)
    
    t_sites = list()
    for i in range(ND):
        for j in range(ND):
            for k in range(ND):
                t_sites.append([dx[i],dy[j],dz[k]])
             
    #print t_site
    
    source_sites = np.array(t_sites, dtype=np.dtype('d'))
    epsilons = np.array([math.sqrt(2),], dtype=np.dtype('d'))
    
    gmls_obj.setWindowSizes(epsilons)
    gmls_obj.setSourceSites(source_sites)
    gmls_obj.setTargetSites(target_sites)
    gmls_obj.setNeighbors(neighbor_lists)
    
    gmls_obj.generatePointEvaluationStencil()
    
    np_data_sources = np.array([1.0]*1000, dtype=np.dtype('d'))
    print gmls_obj.applyStencilTo0Tensor(0, neighbor_lists, np_data_sources)
    
    print gmls_obj

def test2():
    
    # 3rd order reconstruction using 2nd order basis in 1D manifold embeded in 2D
    gmls_manifold_obj=gmls.GMLS_Python(2, "MANIFOLD", 2, 2)
    gmls_manifold_obj.setWeightingOrder(8,8)
    
    if (len(sys.argv)>1):
        ND = int(sys.argv[1])
    else:
        ND = 200

    if (len(sys.argv)>2):
        xmax = float(sys.argv[2])
    else:
        xmax = 0.1

    xmin = -xmax

    
    neighbor_lists = np.array([[ND,] + range(ND)], dtype=np.dtype(int))
    target_sites = np.array([[0,1],], dtype=np.dtype('d'))
    
    nx, ny = (ND,ND,)
    dx = np.linspace(xmin, xmax, nx)
    dy = 1-np.power(dx,2)/math.pow(xmin,2) # parabolic
    
    t_sites = list()
    for i in range(ND):
        t_sites.append([dx[i],dy[i]])
             
    #print t_site
    
    source_sites = np.array(t_sites, dtype=np.dtype('d'))
    epsilons = np.array([abs(xmax)*math.sqrt(2),], dtype=np.dtype('d'))
    
    gmls_manifold_obj.setWindowSizes(epsilons)
    gmls_manifold_obj.setSourceSites(source_sites)
    gmls_manifold_obj.setTargetSites(target_sites)
    gmls_manifold_obj.setNeighbors(neighbor_lists)
    
    gmls_manifold_obj.generatePointEvaluationStencil()
    
    # todo
    #np_data_sources = np.array([1.0]*1000, dtype=np.dtype('d'))
    np_data_sources = np.cos(dx)*np.cos(dy)
    ans = gmls_manifold_obj.applyStencilTo0Tensor(0, neighbor_lists, np_data_sources)

    print ans, np.cos(1), ans-np.cos(1)
    #print ans, 1, ans-1
    print gmls_manifold_obj
    #plt.plot(dx, np_data_sources)
    #plt.plot([0], [ans], marker='o', markersize=3, color="red")
    #plt.show()

test1()
test2()
