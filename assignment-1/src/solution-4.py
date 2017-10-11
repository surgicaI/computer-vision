import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sfm = sio.loadmat('res/sfm_points.mat')

image_points = sfm['image_points']

#Calculate centroid and subtract from image points
for i in range(0,10):
    x_centroid=sum(image_points[0,:,i])/len(image_points[0,:,i])
    y_centroid=sum(image_points[1,:,i])/len(image_points[1,:,i])
    image_points[0,:,i] = image_points[0,:,i] - x_centroid
    image_points[1,:,i] = image_points[1,:,i] - y_centroid

    #For first camera
    if i==0:
        print("T for first camera")
        print(x_centroid)
        print(y_centroid)


#Creating 2mxn Matrix
w = image_points[:,:,0]
for i in range(1,10):
    w=np.vstack((w,image_points[:,:,i]))


#SVD of W
u,s,v=np.linalg.svd(w)

#M for first camera
M=np.dot(u[:,0:3],np.array([[s[0],0,0],[0,s[1],0],[0,0,s[2]]]))
print("M for first camera")
print(M[0:2])


#Printing 10 coordinates
print("First 10 coordinates")
print(np.transpose(v[0:3])[0:10])

#Below code takes some time .. message for the same
print("It take some time to show up ..")

#Plotting coordinates
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0,600):
    ax.scatter(v[0][i], v[1][i], v[2][i])

#displaying output
plt.show()
