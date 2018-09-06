import math
# creates a wavy grid


content=[]

content.append("# vtk DataFile Version 2.0")
content.append("particlePositions")
content.append("ASCII")
content.append("DATASET POLYDATA")

h=0.0125

height = 1.0
width = 1.0

z_layer = int(height / h)
# arclength of .1*sin(5 pi x) \approx 1.4637
x_layer = int(1.4637 * width / h)


points = []

for vertical in range(z_layer):
    for horizontal in range(x_layer):
        x = horizontal * (height / (x_layer-1))
        y = 0.1*math.sin(5*math.pi*x)
        z = vertical * (height / (z_layer-1))
        if vertical == 0 or vertical == z_layer-1 or horizontal == 0 or horizontal == x_layer-1:
            points.append([x,y,z,1])
        else:
            points.append([x,y,z,0])
         

content.append("POINTS %d float"%(z_layer*x_layer))
for point in points:
    content.append("%f %f %f"%(point[0],point[1],point[2]))

content.append("POINT_DATA %d"%(z_layer*x_layer))
content.append("SCALARS FLAG float 1")
content.append("LOOKUP_TABLE default")

for point in points:
    content.append("%f"%point[3])

# fake grid_area
content.append("SCALARS grid_area float 1")
content.append("LOOKUP_TABLE default")

for point in points:
    content.append("%f"%(1./(z_layer*x_layer)))

with open("wavy.vtk",'w') as file:
    for line in content:
        file.write(line+'\n')

print content
