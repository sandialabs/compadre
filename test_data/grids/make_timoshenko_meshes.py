# with examples from git@gitlab.sandia.gov:jkoeste/example_problems.git cloned into ../example_problems
import sys
sys.path.append("../utilities")
import convert_genesis_to_compadre

# standard regular
grid_nums=range(5)
input_names = ["../example_problems/timoshenko_beam/regular/mesh/mesh_"+str(gnum)+".g" for gnum in grid_nums]
output_names = ["timoshenko_regular_mesh_"+str(gnum)+".nc" for gnum in grid_nums]
for key,fname_in in enumerate(input_names):
    convert_genesis_to_compadre.convert(fname_in, output_names[key], "separate", "coordx", "coordy", "", "connect1", "num_nod_per_el1", 2, 1.0, "true", "false")
# scaled irregular
output_names = ["timoshenko_regular_mesh_scaled_"+str(gnum)+".nc" for gnum in grid_nums]
for key,fname_in in enumerate(input_names):
    convert_genesis_to_compadre.convert(fname_in, output_names[key], "separate", "coordx", "coordy", "", "connect1", "num_nod_per_el1", 2, 1.0/24.0, "true", "false")

# standard irregular
grid_nums=range(4)
input_names = ["../example_problems/timoshenko_beam/irregular/mesh/mesh_"+str(gnum)+".g" for gnum in grid_nums]
output_names = ["timoshenko_irregular_mesh_"+str(gnum)+".nc" for gnum in grid_nums]
for key,fname_in in enumerate(input_names):
    convert_genesis_to_compadre.convert(fname_in, output_names[key], "separate", "coordx", "coordy", "", "connect1", "num_nod_per_el1", 2, 1.0, "true", "false")
# scaled irregular
output_names = ["timoshenko_irregular_mesh_scaled_"+str(gnum)+".nc" for gnum in grid_nums]
for key,fname_in in enumerate(input_names):
    convert_genesis_to_compadre.convert(fname_in, output_names[key], "separate", "coordx", "coordy", "", "connect1", "num_nod_per_el1", 2, 1.0/24.0, "true", "false")

# standard large_angle
grid_nums=range(5)
input_names = ["../example_problems/timoshenko_beam/large_angle/mesh/mesh_"+str(gnum)+".g" for gnum in grid_nums]
output_names = ["timoshenko_large_angle_mesh_"+str(gnum)+".nc" for gnum in grid_nums]
for key,fname_in in enumerate(input_names):
    convert_genesis_to_compadre.convert(fname_in, output_names[key], "separate", "coordx", "coordy", "", "connect1", "num_nod_per_el1", 2, 1.0, "true", "false")
# scaled large_angle
output_names = ["timoshenko_large_angle_mesh_scaled_"+str(gnum)+".nc" for gnum in grid_nums]
for key,fname_in in enumerate(input_names):
    convert_genesis_to_compadre.convert(fname_in, output_names[key], "separate", "coordx", "coordy", "", "connect1", "num_nod_per_el1", 2, 1.0/24.0, "true", "false")
