import numpy as np

# Number of points along each axis
N = 48

# Minimum and maximum value for coordinates
min_coord, max_coord = -1.0, 1.0

# Generate the coordinates
X = np.linspace(min_coord, max_coord, N)
Y = np.linspace(min_coord, max_coord, N)
Z = np.linspace(min_coord, max_coord, N)

# Write to text files
output_file = open("output_{}.vtk".format(N), "w")

# Write the header
output_file.write("# vtk DataFile Version 2.0\n")
output_file.write("particlePositions\n")
output_file.write("ASCII\n")
output_file.write("DATASET POLYDATA\n")
output_file.write("POINTS {} float\n".format(N*N*N))

# Loop through each points in the coordinate
for x in X:
    for y in Y:
        for z in Z:
            output_file.write("{:.6f} {:.6f} {:.6f}\n".format(x, y, z))

output_file.write("POINT_DATA %d\n"%(N*N*N))
output_file.write("SCALARS FLAG float 1\n")
output_file.write("LOOKUP_TABLE default\n")

for x in X:
    for y in Y:
        for z in Z:
            if ((x==X[0] or x==X[-1]) or (y==Y[0] or y==Y[-1]) or (z==Z[0] or z==Z[-1])):
                output_file.write("{:d}\n".format(1))
            else:
                output_file.write("{:d}\n".format(0))

output_file.write("SCALARS P float 1\n")
output_file.write("LOOKUP_TABLE default\n")

for x in X:
    for y in Y:
        for z in Z:
            output_file.write("{:.1f}\n".format(0))


output_file.close()
