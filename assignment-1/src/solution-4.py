import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sfm = sio.loadmat('res/sfm_points.mat')
LOGS_ENABLED = False

image_points = sfm['image_points']

# Compute the translations t
# i directly by computing the centroid of point
# in each image i.
# Center the points in each image by subtracting off the centroid, so that
# the points have zero mean
for i in range(0,10):
    x_centroid=sum(image_points[0,:,i])/len(image_points[0,:,i])
    y_centroid=sum(image_points[1,:,i])/len(image_points[1,:,i])
    image_points[0,:,i] = image_points[0,:,i] - x_centroid
    image_points[1,:,i] = image_points[1,:,i] - y_centroid

    # The script should print out the Mi and ti
    # for the first camera
    if i==0:
        print("T for first camera")
        print('x-centroid', x_centroid)
        print('y-centroid', y_centroid)


# Construct the 2m by n measurement matrix W from the centered data.
w = image_points[:,:,0]
for i in range(1,10):
    w=np.vstack((w,image_points[:,:,i]))

# Perform an SVD decomposition of W into UDV T
u,s,v=np.linalg.svd(w)

# The camera locations Mi
# can be obtained from the first three columns
# of U multiplied by D(1 : 3, 1 : 3), the first three singular values
M=np.dot(u[:,0:3],np.array([[s[0],0,0],[0,s[1],0],[0,0,s[2]]]))
# The script shouldprint out the Mi and ti
# for the first camera
print("M for first camera")
print(M[0:2])


# The script should print out the Mi and ti
# for the first camera and also the 3D coordinates of
# the first 10 world points.
print("First 10 coordinates")
print(np.transpose(v[0:3])[0:10])

# You can verify your answer by plotting the 3D world points out. using
# the plot3 command. The rotate3d command will let you rotate the
# plot. This functionality is replicated in Python within the matplotlib
# package
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0,600):
    ax.scatter(v[0][i], v[1][i], v[2][i])

#displaying output
plt.show()
