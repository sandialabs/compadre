import os
import sys
import numpy as np
from netCDF4 import Dataset

# to be run from ./build/examples
# gets many instances of backward_i.pvtp.g, forward_i.pvtp.g, and concatenates
# them into a combo file

def copy_variables_from_source_except_data(filename_source, dataset, new_fields):
    f = Dataset(filename_source, "r", format="NETCDF4")
    dimensions = f.dimensions
    variables = f.variables

    for attname in f.ncattrs():
        setattr(dataset,attname,getattr(f,attname))
    
    # To copy the dimension of the netCDF file
    for dimname,dim in f.dimensions.items():
        dataset.createDimension(dimname,len(dim))

    # To copy the variables of the netCDF file
    for varname,ncvar in f.variables.items():
        list_of_keys = list(new_fields.keys())
        if (varname not in list_of_keys):
            var = dataset.createVariable(varname,ncvar.dtype,ncvar.dimensions)
            var[:] = ncvar[:]
        else:
            if (varname != "ID"):
                dataset.createVariable(varname, datatype='d', dimensions=('num_el_in_blk1','time_step',), zlib=False, complevel=4,\
                                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                                       endian='native', least_significant_digit=None, fill_value=None)
                #print(new_fields[varname].shape)
                #print(dataset.variables[varname][:].shape)
            else:
                dataset.createVariable(varname, datatype='i8', dimensions=('num_el_in_blk1',), zlib=False, complevel=4,\
                                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                                       endian='native', least_significant_digit=None, fill_value=None)
            dataset.variables[varname][:]=new_fields[varname]

    f.close()

def get_data_from_file_sequence(iters, file_prefix, fields):
    concatenated_data = {}
    for field in fields:
        concatenated_data[field]=np.zeros(shape=(0,0),dtype='d')

    for i in range(iters):
        f = Dataset(file_prefix+str(i+1)+".pvtp.g", "r", format="NETCDF4")
        for varname,ncvar in f.variables.items():
            if (varname in fields):
                if (varname != "ID"):
                    if (concatenated_data[varname].shape[0]==0):
                        concatenated_data[varname]=np.zeros(shape=(ncvar.shape[0],iters), dtype='d')
                        concatenated_data[varname][:,0]=ncvar[:].flatten()
                    else:
                        concatenated_data[varname][:,i] = ncvar[:].flatten()
                elif (varname=="ID" and i==1):
                    # take care of ID here on first step
                    concatenated_data[varname]=np.zeros(shape=(ncvar.shape[0],), dtype='d')
                    concatenated_data[varname][:]=ncvar[:].flatten()

    return concatenated_data

def consolidate(iters, file1, file2):

    head, tail = os.path.split(file1)
    file1_short = tail

    head, tail = os.path.split(file2)
    file2_short = tail

    # new output file name
    new_filename = file1_short + "-" + file2_short

    # get data from various data files
    field_dictionary = get_data_from_file_sequence(iters, "backward_", ["ID","TotalPrecipWater","CloudFraction","Topography"])

    # create an empty dataset
    dataset = Dataset(new_filename, "w", format="NETCDF4")

    # fill in dataset from source file
    # but not copying TotalPrecWater, etc....
    copy_variables_from_source_except_data(file1, dataset, field_dictionary)

    # close file we are writing to
    dataset.close()

    # all steps need duplicated, and called with "forward_" with file1+2 reversed
    head, tail = os.path.split(file1)
    file1_short = tail

    head, tail = os.path.split(file2)
    file2_short = tail

    # new output file name
    new_filename = file2_short + "-" + file1_short

    # get data from various data files
    field_dictionary = get_data_from_file_sequence(iters, "forward_", ["ID","TotalPrecipWater","CloudFraction","Topography"])

    # create an empty dataset
    dataset = Dataset(new_filename, "w", format="NETCDF4")

    # fill in dataset from source file
    # but not copying TotalPrecWater, etc....
    copy_variables_from_source_except_data(file2, dataset, field_dictionary)

    # close file we are writing to
    dataset.close()


#test_file_1 = "../../test_data/grids/canga/Cubed-Sphere/outCSMesh_ne16_TPW_CFR_TPO.g"
#test_file_2 = "../../test_data/grids/canga/Cubed-Sphere/outCSMesh_ne16_TPW_CFR_TPO.g"
#consolidate(10,test_file_1,test_file_2)

