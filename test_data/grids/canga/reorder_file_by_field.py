import os
import sys
import numpy as np
from netCDF4 import Dataset

ordering_field = "ID"
suffix_names = ["_remap_src", "_remap_tgt"]

# to be run from ./build/examples
# gets a file by name, orders by "ordering_field" 

def reorder_file_by_field(filename_source, suffix):
    f = Dataset(filename_source, "r+", format="NETCDF4")

    # get the ordering field
    filtering_dimension = ""
    unsorted_ids = None
    for varname,ncvar in f.variables.items():
        if (varname==ordering_field+suffix):
            # should only have one dimension
            dimensions = list(ncvar.dimensions[:])
            filtering_dimension = dimensions[0]
            unsorted_ids = ncvar[:]
            break
    assert filtering_dimension!="", "%s field not found"%(ordering_field+suffix)
    unsorted_ids = np.array(unsorted_ids, dtype='int')

    #print(unsorted_ids)
    reordering_to_apply = np.argsort(unsorted_ids, axis=-1, kind='mergesort')
    #print(reordering_to_apply)
    #print(unsorted_ids[reordering_to_apply])

    # use the reordering on fields relying on same dimension
    for varname,ncvar in f.variables.items():
        # check first dimension
        dimensions = list(ncvar.dimensions[:])
        if (dimensions[0]==filtering_dimension):
            # assumed column zero has ordered data
            if (len(dimensions)>1):
                tmp = ncvar[reordering_to_apply,1:]
                ncvar[:,1:] = tmp
            elif (varname==ordering_field+suffix):
                tmp = ncvar[reordering_to_apply][:]
                ncvar[:] = tmp
            else:
                pass
                # field is from original data file (that is how consolidate.py works)
                # and shouldn't be reordered, since it was never mixed up

    f.close()

if __name__ == '__main__':
    file_name_to_reorder = ""
    if (len(sys.argv) > 1):
        file_name_to_reorder = str(sys.argv[1])
    for suffix in suffix_names:
        reorder_file_by_field(file_name_to_reorder, suffix)



#test_file_1 = "../../test_data/grids/canga/Cubed-Sphere/outCSMesh_ne16_TPW_CFR_TPO.g"
#test_file_2 = "../../test_data/grids/canga/Cubed-Sphere/outCSMesh_ne16_TPW_CFR_TPO.g"
#consolidate(10,test_file_1,test_file_2)

