import cv2, os, copy
import numpy as np

# reading input images and saving them respectively to image1 and image2
image1  = cv2.imread('res/scene.pgm')
image2  = cv2.imread('res/book.pgm')

# computing sift keypoints
sift = cv2.xfeatures2d.SIFT_create()
kp_image1, des_image1 = sift.detectAndCompute(image1,None)
kp_image2, des_image2 = sift.detectAndCompute(image2,None)

image1_plain = copy.deepcopy(image1)
image2_plain = copy.deepcopy(image2)
# drawing sift keypoints on both the images
cv2.drawKeypoints(image1, kp_image1, image1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(image2, kp_image2, image2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
file_1 = 'out/sift_keypoints_image_1.jpg'
file_2 = 'out/sift_keypoints_image_2.jpg'
out_dir = 'out'
if not os.path.isdir(out_dir):
    print('creating dir:' + out_dir)
    os.makedirs(out_dir)
if not os.path.isfile(file_1):
    print('writing image1 with keypoints at path:'+file_1)
    cv2.imwrite(file_1, image1)
if not os.path.isfile(file_2):
    print('writing image2 with keypoints at path:'+file_2)
    cv2.imwrite(file_2, image2)


# Obtaining a set of putative matches T using opencv2 Brute force matcher.
# BFMatcher documentation: http://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html
myBFMatcher = cv2.BFMatcher(cv2.NORM_L2)
matches = myBFMatcher.knnMatch(des_image1, des_image2, k=2)

# Remove the spurious matches with threshold = 0.9
threshold = 0.9
result_matches = []
for a,b in matches:
    if a.distance < threshold * b.distance:
        result_matches.append(a)

# To check that your code is functioning correctly,
# plot out the two images side-by-side with lines
# showing the potential matches
imageWithMatches = copy.deepcopy(image1_plain)
imageWithMatches = cv2.drawMatches(image1_plain, kp_image1, image2_plain, kp_image2, result_matches, imageWithMatches, flags=0)
file_3 = 'out/feature_matching.jpg'
if not os.path.isfile(file_3):
    print('writing imageWithMatches with keypoints at path:'+file_3)
    cv2.imwrite(file_3, imageWithMatches)
