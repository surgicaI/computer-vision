import cv2
import numpy as np

img_1  = cv2.imread('res/scene.pgm')

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img_1,None)

cv2.drawKeypoints(img_1, kp, img_1)

cv2.imwrite('out/sift_keypoints.jpg',img_1)
