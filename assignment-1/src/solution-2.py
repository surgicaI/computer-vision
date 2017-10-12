import cv2, os, copy, random, math
import numpy as np

# generates 3 random numbers from 0 to n-1
def generateRandomNums(n):
    result = set()
    while len(result) < 3:
        m = random.randrange(n)
        if not m in result:
            result.add(m)
    return list(result)

LOGS_ENABLED = False

# reading input images and saving them respectively to image1 and image2
image1  = cv2.imread('res/book.pgm')
image2  = cv2.imread('res/scene.pgm')

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
num_inliers_max = 0
for i in range(N):
    x, y, z = generateRandomNums(len(result_matches))
    if LOGS_ENABLED:
        print(x, y, z)
    P_matches = [result_matches[x], result_matches[y], result_matches[z]]

    #Matrix Construction  A and b
    A = np.zeros(shape=(6,6))
    b = np.zeros(shape=(6,1))
    for i in range(P):
        desc1_idx = P_matches[i].queryIdx
        desc2_idx = P_matches[i].trainIdx
        kp1, kp2 = kp_image1[desc1_idx].pt, kp_image2[desc2_idx].pt
        A[2*i][0]=kp1[0]
        A[2*i][1]=kp1[1]
        A[2*i][4]=1
        A[2*i+1][2]=kp1[0]
        A[2*i+1][3]=kp1[1]
        A[2*i+1][5]=1
        b[2*i][0]=kp2[0]
        b[2*i+1][0]=kp2[1]

    # Solve for the unknown transformation parameters q. In Matlab you
    # can use the \ command. In Python you can use linalg.solve.
    try:
        q = np.linalg.solve(A,b)
    except np.linalg.LinAlgError:
        print('A is singular')
        continue

    if LOGS_ENABLED:
        print(q)

    # Using the transformation parameters, transform the locations of all T
    # points in image 1. If the transformation is correct, they should lie close
    # to their pairs in image 2.
    # Count the number of inliers, inliers being defined as the number of
    # transformed points from image 1 that lie within a radius of 10 pixels
    # of their pair in image 2.
    num_inliers=0
    inliers=[]
    for match in result_matches:
        kp1 = kp_image1[match.queryIdx].pt
        kp2 = kp_image2[match.trainIdx].pt
        x = np.zeros(shape=(2,6))
        x[0][0]=kp1[0]
        x[0][1]=kp1[1]
        x[0][4]=1
        x[1][2]=kp1[0]
        x[1][3]=kp1[1]
        x[1][5]=1
        R=np.dot(x,q)
        if(math.hypot(R[0][0] - kp2[0], R[1][0] - kp2[1])<10):
            num_inliers = num_inliers + 1
            inliers.append([kp1,kp2])

    # If this count exceeds the best total so far, save the transformation
    # parameters and the set of inliers.
    if(num_inliers >= num_inliers_max):
        transformation = q
        inliers_set = inliers
        num_inliers_max = num_inliers

# Perform a final refit using the set of inliers belonging to the best transformation
# you found. This refit should use all inliers, not just 3 points
# chosen at random.
A = np.zeros(shape=(2*len(inliers_set),6))
b = np.zeros(shape=(2*len(inliers_set),1))
for i in range(0,len(inliers_set)):
    kp1,kp2 = inliers_set[i]
    A[2*i][0] = kp1[0]
    A[2*i][1] = kp1[1]
    A[2*i][4] = 1
    A[2*i+1][2] = kp1[0]
    A[2*i+1][3] = kp1[1]
    A[2*i+1][5] = 1
    b[2*i][0] = kp2[0]
    b[2*i+1][0] = kp2[1]
q,residuals,rank,s = np.linalg.lstsq(A,b)
transformation = q

np.savetxt('out/image-alignment-transformation', transformation, delimiter=',')

#Making homography matrix
H = np.zeros(shape=(2,3))
H[0][0]=transformation[0][0]
H[0][1]=transformation[1][0]
H[0][2]=transformation[4][0]
H[1][0]=transformation[2][0]
H[1][1]=transformation[3][0]
H[1][2]=transformation[5][0]
image1  = cv2.imread('res/book.pgm')
rows, cols, _ = image1.shape
affine_transform = cv2.warpAffine(image1,H,(rows,cols))
cv2.imwrite('out/affine_transform.jpg',affine_transform)
