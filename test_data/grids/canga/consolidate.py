import os
import sys
import numpy as np
from netCDF4 import Dataset

# to be run from ./build/examples
# gets many instances of backward_i.g, forward_i.g, and concatenates
# them into a combo file

def copy_variables_from_source_except_data(filename_source, dataset, new_fields, suffix):
    f = Dataset(filename_source, "r", format="NETCDF4")
    dimensions = f.dimensions
    variables = f.variables

    for attname in f.ncattrs():
        setattr(dataset,attname,getattr(f,attname))
    
    # To copy the dimension of the netCDF file
    for dimname,dim in f.dimensions.items():
        if (dimname=="time_step"):
            if (dimname not in list(dataset.dimensions.keys())):
                dataset.createDimension(dimname,len(dim))
        else:
            dataset.createDimension(dimname+suffix,len(dim))

    # loop new_fields, add a pair with a boolean for whether added
    new_fields_added = set()

    # To copy the variables of the netCDF file
    for varname,ncvar in f.variables.items():
        list_of_keys = list(new_fields.keys())
        if (varname not in list_of_keys):
            dimensions = list(ncvar.dimensions[:])
            for key,item in enumerate(dimensions):
                if item=="time_step":
                    dimensions[key] = item
                else:
                    dimensions[key] = item + suffix
            var = dataset.createVariable(varname+suffix,ncvar.dtype,dimensions)
            var[:] = ncvar[:]
        else:
            # catches when field already exists, but we change dimension
            if (varname != "ID"):
                dataset.createVariable(varname+suffix, datatype='f8', dimensions=('num_el_in_blk1'+suffix,'time_step',), zlib=False, complevel=4,\
                                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                                       endian='native', least_significant_digit=None, fill_value=None)
                #print(new_fields[varname].shape)
                #print(dataset.variables[varname][:].shape)
            else:
                dataset.createVariable(varname+suffix, datatype='i8', dimensions=('num_el_in_blk1'+suffix,), zlib=False, complevel=4,\
                                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                                       endian='native', least_significant_digit=None, fill_value=None)
            dataset.variables[varname+suffix][:]=new_fields[varname]
        new_fields_added.add(varname)

    # add all fields not existing in original file
    for varname in list(new_fields.keys()):
        if varname not in new_fields_added:
            if (varname != "ID"):
                dataset.createVariable(varname+suffix, datatype='f8', dimensions=('num_el_in_blk1'+suffix,'time_step',), zlib=False, complevel=4,\
                                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                                       endian='native', least_significant_digit=None, fill_value=None)
            else:
                dataset.createVariable(varname+suffix, datatype='i8', dimensions=('num_el_in_blk1'+suffix,), zlib=False, complevel=4,\
                                       shuffle=True, fletcher32=False, contiguous=False, chunksizes=None,\
                                       endian='native', least_significant_digit=None, fill_value=None)
            dataset.variables[varname+suffix][:]=new_fields[varname]

    f.close()

def filter_fields_in_file_sequence(iters, reference_file, file_prefix, fields):
    # make sure all requested fields exist in file sequence (e.g. Smooth may not)
    updated_fields = list()
    f = Dataset(file_prefix+str(1)+".g", "r", format="NETCDF4")
    for field_name in fields:
        for varname,ncvar in f.variables.items():
            if (field_name == varname):
                updated_fields.append(field_name)
                break
    f.close()
    return updated_fields

def get_data_from_file_sequence(iters, reference_file, file_prefix, fields):
    concatenated_data = {}
    for field in fields:
        if (field=="ID"):
            concatenated_data[field]=np.zeros(shape=(0,0),dtype='i8')
        else:
            concatenated_data[field]=np.zeros(shape=(0,0),dtype='f8')

    for i in range(iters):
        f = Dataset(file_prefix+str(i+1)+".g", "r", format="NETCDF4")
        for varname,ncvar in f.variables.items():
            if (varname in fields):
                if (varname != "ID"):
                    #print("%s, %s time_step: %d"%(file_prefix, varname, i))
                    if (concatenated_data[varname].shape[0]==0):
                        concatenated_data[varname]=np.zeros(shape=(ncvar.shape[0],iters+1), dtype='f8')
                        concatenated_data[varname][:,1]=ncvar[:].flatten()
                    else:
                        concatenated_data[varname][:,i+1] = ncvar[:].flatten()
                elif (varname=="ID" and i==1):
                    # take care of ID here on first step
                    concatenated_data[varname]=np.zeros(shape=(ncvar.shape[0],), dtype='i8')
                    concatenated_data[varname][:]=ncvar[:].flatten()
        f.close()

    # if reference_file didn't have this field, then it will have zeros for that time_step
    ref = Dataset(reference_file, "r", format="NETCDF4")
    for varname,ncvar in ref.variables.items():
        if (varname in fields):
            if (varname != "ID"):
                concatenated_data[varname][:,0]=ncvar[:].flatten()
    ref.close()

    return concatenated_data

def consolidate(iters, file1, file2):

    original_field_names = ["ID","TotalPrecipWater","CloudFraction","Topography","Smooth","AnalyticalFun1","AnalyticalFun2"]
    field_names = filter_fields_in_file_sequence(iters, file1, "forward_", original_field_names)

    head, tail = os.path.split(file1)
    file1_short = tail

    head, tail = os.path.split(file2)
    file2_short = tail

    # new output file name
    new_filename = file1_short + "-" + file2_short

    # get data from various data files
    field_dictionary = get_data_from_file_sequence(iters, file1, "backward_", field_names)

    # create an empty dataset
    dataset1 = Dataset(new_filename, "w", format="NETCDF4")

    # fill in dataset from source file
    # but not copying TotalPrecWater, etc....
    copy_variables_from_source_except_data(file1, dataset1, field_dictionary, "_remap_src")

    # close file we are writing to
    #dataset1.close()

    # all steps need duplicated, and called with "forward_" with file1+2 reversed
    head, tail = os.path.split(file1)
    file1_short = tail

    head, tail = os.path.split(file2)
    file2_short = tail

    # new output file name
    #new_filename = file2_short + "-" + file1_short

    # get data from various data files
    field_dictionary = get_data_from_file_sequence(iters, file2, "forward_", field_names)

    # create an empty dataset
    #dataset2 = Dataset(new_filename, "w", format="NETCDF4")

    # fill in dataset from source file
    # but not copying TotalPrecWater, etc....
    copy_variables_from_source_except_data(file2, dataset1, field_dictionary, "_remap_tgt")

    # close file we are writing to
    dataset1.close()

    return new_filename

if __name__ == '__main__':
    iters=0
    if (len(sys.argv) > 1):
        iters = int(sys.argv[1])
    file1=""
    if (len(sys.argv) > 2):
        file1 = str(sys.argv[2])
    file2=""
    if (len(sys.argv) > 3):
        file2 = str(sys.argv[3])
    consolidate(iters, file1, file2)

#test_file_1 = "../../test_data/grids/canga/Cubed-Sphere/outCSMesh_ne16_TPW_CFR_TPO.g"
#test_file_2 = "../../test_data/grids/canga/Cubed-Sphere/outCSMesh_ne16_TPW_CFR_TPO.g"
#consolidate(10,test_file_1,test_file_2)

