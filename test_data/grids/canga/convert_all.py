import convert_CubedSphere_add_ID_and_enumerate_coords as convert_CS
import convert_CVT_nodes_to_cell_centers as convert_ICOD
import convert_RLL

#grid_nums = [16,32,64,128,256]
#
#input_names = ["original_NM16/sample_NM16_O10_CS-r"+str(gnum)+"_TPW_CFR_TPO_A1_A2.nc" for gnum in grid_nums]
#output_names = ["NM16/CS_"+str(gnum)+".g" for gnum in grid_nums]
#for key,fname_in in enumerate(input_names):
#    convert_CS.convert(fname_in,output_names[key])
#
#input_names = ["original_NM16/sample_NM16_O10_ICOD-r"+str(gnum)+"_TPW_CFR_TPO_A1_A2.nc" for gnum in grid_nums]
#output_names = ["NM16/ICOD_"+str(gnum)+".g" for gnum in grid_nums]
#for key,fname_in in enumerate(input_names):
#    convert_ICOD.convert(fname_in,output_names[key])

deg_nums = ['30-60','90-180','180-360','360-720','720-1440']
out_deg_nums = ['30-60','90-180','180-360','360-720','720-1440']
input_names = ["original_NM16/sample_NM16_O10_RLL-r"+dnum+"_TPW_CFR_TPO_A1_A2.nc" for dnum in deg_nums]
output_names = ["NM16/RLL_"+dnum+".g" for dnum in out_deg_nums]
for key,fname_in in enumerate(input_names):
    # convert_CS does same thing that convert_RLL would do
    print("working on:",output_names[key])
    convert_RLL.convert(fname_in,output_names[key])
print("complete.")
