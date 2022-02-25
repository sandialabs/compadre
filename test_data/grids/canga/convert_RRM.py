import convert_CubedSphere_add_ID_and_enumerate_coords as convert_CS
import convert_RRM_CVT as convert_ICOD

grid_nums = [32,64,128]

#input_names = ["original_RRM/sample_NM32_O18_CS-r"+str(gnum)+"_TPW_CFR_TPO_A1_A2.nc" for gnum in grid_nums]
#output_names = ["NM32/RRM_CS_"+str(gnum)+".g" for gnum in grid_nums]
#for key,fname_in in enumerate(input_names):
#    convert_CS.convert(fname_in,output_names[key])

input_names = ["original_RRM/sample_NM32_O18_ICOD-r"+str(gnum)+"_TPW_CFR_TPO_A1_A2.nc" for gnum in grid_nums]
output_names = ["NM32/RRM_ICOD_"+str(gnum)+".g" for gnum in grid_nums]
for key,fname_in in enumerate(input_names):
    convert_ICOD.convert(fname_in,output_names[key])

