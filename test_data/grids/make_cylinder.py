import math
import random



#h=0.2 is 0
#h=0.1 is 1
#h=0.05 is 2
#h=0.0250 is 3
#h=0.0125 is 4
#h=0.00625 is 5
h_all=[0.2,0.1,0.05,0.025,0.0125,0.00625]

height = 2.0
radius = 1.0

variation = .00#15 # as a decimal for a percent


for key, h in enumerate(h_all):
    content=[]
    
    content.append("# vtk DataFile Version 2.0")
    content.append("particlePositions")
    content.append("ASCII")
    content.append("DATASET POLYDATA")
    
    circumference_points = int(2*math.pi*radius / h)
    z_layer = int(height / h)
    flagged_layers_top_and_bottom = .00001*z_layer
    
    angle_delta = 2*math.pi/circumference_points
    
    points = []
    
    for vertical in range(z_layer):
        for horizontal in range(circumference_points):
            rand_pert_x = random.uniform(-variation, variation)
            rand_pert_y = random.uniform(-variation, variation)
            rand_pert_z = random.uniform(-variation, variation)
    
            x = math.sin(horizontal*angle_delta) + h*rand_pert_x
            y = math.cos(horizontal*angle_delta) + h*rand_pert_y
            #x = math.sin(horizontal*angle_delta+math.pi/2) + h*rand_pert_x
            #y = math.cos(horizontal*angle_delta+math.pi/2) + h*rand_pert_y
            
            # move back onto cylinder
            norm = math.sqrt(pow(x,2) + pow(y,2))
            x = x / norm
            y = y / norm
     
            baseline_vert = 0 if vertical==0 else (vertical-1) * (height / (z_layer-1))
            z = vertical*(height / (z_layer-1))
            #if (vertical-flagged_layers_top_and_bottom < 0) or 
            if (vertical-flagged_layers_top_and_bottom < 0) or (vertical - (z_layer-1 - flagged_layers_top_and_bottom) > 0) or horizontal < circumference_points/2:
                points.append([x,y,z,1])
            else:
                points.append([x,y,z+rand_pert_z*h,0]) # don't move boundary pts irregularly
    
    #        if vertical == 0 or vertical == z_layer-1:
    #            points.append([x,y,z,1])
    #        else:
    #            points.append([x,y,z,0])
             
    
    content.append("POINTS %d float"%(z_layer*circumference_points))
    for point in points:
        content.append("%f %f %f"%(point[0],point[1],point[2]))
    
    content.append("POINT_DATA %d"%(z_layer*circumference_points))
    content.append("SCALARS FLAG float 1")
    content.append("LOOKUP_TABLE default")
    
    for point in points:
        content.append("%f"%point[3])
    
    # fake grid_area
    #content.append("POINT_DATA %d"%(z_layer*circumference_points))
    content.append("SCALARS grid_area float 1")
    content.append("LOOKUP_TABLE default")
    
    for point in points:
        content.append("%f"%(1./(z_layer*circumference_points)))
    
    with open("cylinder_%d.vtk"%key,'w') as file:
        for line in content:
            file.write(line+'\n')

#print content
