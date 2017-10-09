import cv2, os, copy, random
import numpy as np

# generates 3 random numbers from 0 to n-1
def generateRandomNums(n):
    result = set()
    while len(result) < 3:
        m = random.randrange(n)
        if not m in result:
            result.add(m)
    return list(result)

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

# RANSAC
N = 100
P = 3
for i in range(N):
    x, y, z = generateRandomNums(len(result_matches))
    print(x, y, z)
    P_matches = [result_matches[x], result_matches[y], result_matches[z]]

    #Matrix Construction  A and b
    A = np.zeros(shape=(6,6))
    b = np.zeros(shape=(6,1))
    for i in range(P):
        kp1, kp2 = P_matches[i]
        A[2*i][0]=kp1[0]
        A[2*i][1]=kp1[1]
        A[2*i][4]=1
        A[2*i+1][2]=kp1[0]
        A[2*i+1][3]=kp1[1]
        A[2*i+1][5]=1
        b[2*i][0]=kp2[0]
        b[2*i+1][0]=kp2[1]

    # Solving A,b using linear algebra api, q is the resultant transformation matrix
    try:
        q = np.linalg.solve(A,b)
    except np.linalg.LinAlgError: #if A is singular
        continue

    # print(q)
