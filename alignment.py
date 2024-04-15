import cv2
import numpy as np

img2 = cv2.imread("/home/ahmed/IMAGES/pic.png")
img1 = cv2.imread("/home/ahmed/IMAGES/scanned.png")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

MAX = 500
orb = cv2.ORB_create(MAX)
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
display1 = cv2.drawKeypoints(img1, keypoints1, outImage = np.array([]), color = (255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
display2 = cv2.drawKeypoints(img2, keypoints2, outImage = np.array([]), color = (255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = list(matcher.match(descriptors1, descriptors2, None))

matches.sort(key=lambda x: x.distance, reverse=False)
numgoodmatches = int(len(matches)*0.1)
matches = matches[:numgoodmatches]

im_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

height, width = img1.shape[:2]
im2_reg = cv2.warpPerspective(img2, h, (width, height))
cv2.imshow("frame1",im2_reg)
cv2.imshow("frame2",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()