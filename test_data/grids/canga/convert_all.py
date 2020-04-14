import convert_CubedSphere_add_ID_and_enumerate_coords as convert_CS
import convert_CVT_nodes_to_cell_centers as convert_ICOD

grid_nums = [16,32,64,128,256]

input_names = ["original_NM16/sample_NM16_O10_CS-r"+str(gnum)+"_TPW_CFR_TPO_A1_A2.nc" for gnum in grid_nums]
output_names = ["NM16/CS_"+str(gnum)+".g" for gnum in grid_nums]
for key,fname_in in enumerate(input_names):
    convert_CS.convert(fname_in,output_names[key])

input_names = ["original_NM16/sample_NM16_O10_ICOD-r"+str(gnum)+"_TPW_CFR_TPO_A1_A2.nc" for gnum in grid_nums]
output_names = ["NM16/ICOD_"+str(gnum)+".g" for gnum in grid_nums]
for key,fname_in in enumerate(input_names):
    convert_ICOD.convert(fname_in,output_names[key])
