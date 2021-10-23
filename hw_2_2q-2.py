# We have to write a function that generates grasp matrix
# Given : Object center and contact locations
# Contact model: Hard finger

import numpy as np
import matplotlib.pyplot as plt

# First the parameters of the object to be grasped is defined
# Measurements are in 'cm'

length = 6
height = 3

# We are asked to increment by 0.1 cm
inc = 0.1 

# Fixing the origin and Object center
# The origin is considered to be fixed at the center of the bottom side of the object
# Hence the object center is calculated to be 
center = np.array([0, height/2])

# We are now provided with four initial forces which are acting on the object
# The location of the contacts points of thee forces are  evaluated with respect to the origin

ci = np.array([[0, 0], [-3, 2], [-1, 3], [1, 3]])

# Let me also define the various metrics function
 
# METRIC - 1               
def minimum_singualr_value(grasp_t):
    w, v = np.linalg.eig(np.matmul(grasp_t.T, grasp_t))
    min_w_sqrt = min(abs(np.sqrt(w)))
    return min_w_sqrt

# METRIC - 2
def ellipsoid_vol(grasp_t):
    det_sqrt = np.sqrt(np.linalg.det(np.matmul(grasp_t.T, grasp_t)))
    return det_sqrt

# METRIC - 3

def grasp_isotropy(grasp_t):
    w, v = np.linalg.eig(np.matmul(grasp_t.T, grasp_t))
    min_w_sqrt = min(abs(np.sqrt(w)))
    max_w_sqrt = max(np.sqrt(w))
    index = min_w_sqrt/max_w_sqrt
    return index    


# Assigning contact frames
# The contact frames for the forces acting on a particular edge is same throught the edge 
# I am hard coding the rotation matrix along the edges
# Since the rotation axis is only along z- axis, the rotation matrix for each case will be 3x3 matrix with z axis as the rotating axis 

top_rotation = np.array([[-1, 0, 0],[0,-1, 0],[0, 0, 1]])
bottom_rotation = np.array([[1, 0, 0], [0, 1 ,0],[0, 0, 1]])
left_rotation = np.array([[0, 1, 0], [-1, 0, 0],[0, 0, 1]])
right_rotation = np.array([[0,-1, 0],[1, 0, 0],[0, 0, 1]])

# Taking transpose because we need rotation matrix that converts coordiantes from base frame to contact frame

top_rotation_t = top_rotation.T
bottom_rotation_t = bottom_rotation.T
left_rotation_t = left_rotation.T
right_rotation_t = right_rotation.T

# h matrix for hard finger model
h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
                  
                 
# calculating the pi matrix with the given contact forces
pi1 = np.array([[1, 0, -(ci[0][1]-center[0])], [0, 1, ci[0][0]-center[1]],  [0, 0, 1]])
pi2 = np.array([[1, 0, -(ci[1][1]-center[0])], [0, 1, ci[1][0]-center[1]],  [0, 0, 1]])
pi3 = np.array([[1, 0, -(ci[2][1]-center[0])], [0, 1, ci[2][0]-center[1]], [0, 0, 1]])
pi4 = np.array([[1, 0, -(ci[3][1]-center[0])], [0, 1, ci[3][0]-center[1]],  [0, 0, 1]])
                
               
# After calculating the respective pi we have to calculate the grasp matrix for the respective contacts or partial grasp matrix and append them to form overall grasp matrix
 
grasp_t = np.matmul(np.matmul(h, bottom_rotation_t), pi1)
grasp_t2 = np.matmul(np.matmul(h, left_rotation_t), pi2)
grasp_t = np.vstack((grasp_t, grasp_t2))
grasp_t3 = np.matmul(np.matmul(h, top_rotation_t), pi3)
grasp_t = np.vstack((grasp_t, grasp_t3))
grasp_t4 = np.matmul(np.matmul(h, top_rotation_t), pi4)
grasp_t = np.vstack((grasp_t, grasp_t4))



# initializing the lists to collect metric result at each sample        


minimum_singualr_values = []
ellipsoid_volumes = []
isotropy_index = []
ref_coordinate = []
x_axis = []


# The iteration across the reactangulor box
# initializing the starting count

count = 0

# Traversing along the top

for i in range(int(length/inc)):
    x = (i - int(length/inc)/2)*inc
    y = int(height/inc)*inc
    #print(x, y)
    pi_i = np.array([[1, 0, -(y-center[0])], [0, 1, x-center[1]],  [0, 0, 1]])
                                     
    grasp_ti = np.matmul(np.matmul(h, top_rotation_t), pi_i)
    grasp_t_new= np.vstack((grasp_t, grasp_ti))
    
    curr_msv = minimum_singualr_value(grasp_t_new)
    minimum_singualr_values.append(curr_msv)
    
    curr_ev = ellipsoid_vol(grasp_t_new)
    ellipsoid_volumes.append(curr_ev)
    
    curr_gi = grasp_isotropy(grasp_t_new)
    isotropy_index.append(curr_gi)
    
    coords = [x, y]
    ref_coordinate.append(coords)
    x_axis.append(count)
    
    count = count + 1

# Traversing along the right side

for i in range(int(height/inc)):
    x = (int(length/inc)/2)*inc
    y = i*inc
    # print(x, y)
    pi_i = np.array([[1, 0, -(y-center[0])], [0, 1, x-center[1]],  [0, 0, 1]])
                    
                   
    grasp_ti = np.matmul(np.matmul(h, right_rotation_t), pi_i)
    grasp_t_new= np.vstack((grasp_t, grasp_ti))
    
    curr_msv = minimum_singualr_value(grasp_t_new)
    minimum_singualr_values.append(curr_msv)
    
    curr_ev = ellipsoid_vol(grasp_t_new)
    ellipsoid_volumes.append(curr_ev)
    
    curr_gi = grasp_isotropy(grasp_t_new)
    isotropy_index.append(curr_gi)
    
    coords = [x, y]
    ref_coordinate.append(coords)
    x_axis.append(count)
    
    count = count + 1

# Traversing across the bottom

for i in range(int(length/inc)):
    x = (i - int(length/inc)/2)*inc
    y = 0
    # print(x, y)
    pi_i = np.array([[1, 0, -(y-center[0])], [0, 1, x-center[1]], [0, 0, 1]])
                                      
    grasp_ti = np.matmul(np.matmul(h, bottom_rotation_t), pi_i)
    grasp_t_new= np.vstack((grasp_t, grasp_ti))
    
    curr_msv = minimum_singualr_value(grasp_t_new)
    minimum_singualr_values.append(curr_msv)
    
    curr_ev = ellipsoid_vol(grasp_t_new)
    ellipsoid_volumes.append(curr_ev)
    
    curr_gi = grasp_isotropy(grasp_t_new)
    isotropy_index.append(curr_gi)
    
    coords = [x, y]
    ref_coordinate.append(coords)
    x_axis.append(count)
    
    count = count + 1

# Traversing across the left side

for i in range(int(height/inc)):
    x = -(int(length/inc)/2)*inc
    y = i*inc
    # print(x, y)
    pi_i = np.array([[1, 0, -(y-center[0])], [0, 1, x-center[1]], [0, 0, 1]])
                    
    grasp_ti = np.matmul(np.matmul(h, left_rotation_t), pi_i)
    grasp_t_new= np.vstack((grasp_t, grasp_ti))
    
    curr_msv = minimum_singualr_value(grasp_t_new)
    minimum_singualr_values.append(curr_msv)
    
    curr_ev = ellipsoid_vol(grasp_t_new)
    ellipsoid_volumes.append(curr_ev)
    
    curr_gi = grasp_isotropy(grasp_t_new)
    isotropy_index.append(curr_gi)
    
    coords = [x, y]
    ref_coordinate.append(coords)
    x_axis.append(count)
    
    count = count + 1

    
# find the maximum of the minimum singular values of all samples
max_minimum_singualr_value = max(minimum_singualr_values)
max_minimum_singualr_value_index = minimum_singualr_values.index(max_minimum_singualr_value)
max_minimum_singualr_value_coord = ref_coordinate[max_minimum_singualr_value_index]
max_ellipsoid_volume = max(ellipsoid_volumes)
max_ellipsoid_volume_index = ellipsoid_volumes.index(max_ellipsoid_volume) 
max_ellipsoid_volume_coord = ref_coordinate[max_ellipsoid_volume_index]
max_isotropy_index = max(isotropy_index)
max_isotropy_index_index = isotropy_index.index(max_isotropy_index)
max_isotropy_index_coord = ref_coordinate[max_isotropy_index_index]

print("minimum singular value:", max_minimum_singualr_value_coord)
print("maximum ellipsoid volume:", max_ellipsoid_volume_coord)
print("maximum isotropy index:", max_isotropy_index_coord)

plt.plot(x_axis, minimum_singualr_values)
plt.show()
