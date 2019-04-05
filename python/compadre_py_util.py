import numpy as np

def get_2D_numpy_array(x, dim_0, dim_1):
    """Converts flat 1D array in 2D numpy array of dimensions dim_0 by dim_1"""
    tmp_array = np.array(x, dtype='d')
    tmp_array = np.reshape(tmp_array,(dim_0,dim_1),'F')
    return tmp_array
