import numpy as np
import scipy.spatial.kdtree as kdtree
import GMLS_Module


def get_neighborlist(source_sites, target_sites, polyOrder, dimensions):
    # neighbor search
    my_kdtree = kdtree.KDTree(source_sites, leafsize=10)
    min_neighbors_needed = GMLS_Module.getNP(polyOrder, dimensions);
    neighbor_lists = []
    temp_neighbor_lists = [] 
    NT = target_sites.shape[0]
    epsilons = np.zeros([NT])

    # first pass specifying number of neighbors needed
    for target_num in range(NT):
        query_result = my_kdtree.query(target_sites[target_num], k=min_neighbors_needed, eps=0)
        epsilons[target_num] = query_result[0][-1]
        assert query_result[1].size >= min_neighbors_needed, "Not enough neighbors exist on this process [%d needed, but %d found]."%(min_neighbors_needed,query_results[1].size)

    # enlarge epsilons by 50%
    # second pass using the enlarged epsilon search
    max_neighbors = 0
    for target_num in range(NT):
        epsilons[target_num] = epsilons[target_num]*1.6;
        query_result = my_kdtree.query_ball_point(target_sites[target_num], epsilons[target_num], p=2.0, eps=0)
        max_neighbors = max(max_neighbors, len(query_result)+1)
        temp_neighbor_lists.append([len(query_result),] + query_result)

    # third pass to pad with 0s
    for target_num in range(NT):
        row = temp_neighbor_lists[target_num]
        row = row + (max_neighbors-len(row))*[0,] # pad with 0s to get the correct dimension
        temp_neighbor_lists[target_num] = row

    # convert list to 2d array
    neighbor_lists = np.array(temp_neighbor_lists, dtype=np.dtype(int))

    d = dict();
    d['neighbor_lists'] = neighbor_lists
    d['epsilons'] = epsilons
    return d

def get_2D_numpy_array(x, dim_0, dim_1):
    tmp_array = np.array(x, dtype='d')
    tmp_array = np.reshape(tmp_array,(dim_0,dim_1),'F')
    return tmp_array
