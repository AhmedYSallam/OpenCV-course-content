import cv2
import numpy as np

img = cv2.imread("/home/ahmed/IMAGES/lemonk.jpg")
img2 = cv2.imread("/home/ahmed/IMAGES/book.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

star = cv2.imread("/home/ahmed/IMAGES/star.png")
circle = cv2.imread("/home/ahmed/IMAGES/circle.png")
graystar = cv2.cvtColor(star, cv2.COLOR_BGR2GRAY)
graycircle = cv2.cvtColor(circle, cv2.COLOR_BGR2GRAY)

mcdo = cv2.imread("/home/ahmed/IMAGES/mcdo.png")
color = cv2.imread("/home/ahmed/IMAGES/color.jpeg")
graymcdo = cv2.cvtColor(mcdo, cv2.COLOR_BGR2GRAY)
graycolor = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
retval, thresh1 = cv2.threshold(gray2, 150, 255, cv2.THRESH_BINARY)
adapthresh = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 49, 20)
adapthresh2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, 20)
#cv2.imshow("mask1", thresh1)
#cv2.imshow("mask2", adapthresh)
#cv2.imshow("mask3", adapthresh2)
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
star_w = graystar.shape[1]
star_h = graystar.shape[0]
aspect_ratio = star_w/graycircle.shape[1]
dim = (star_w, int(graycircle.shape[0]*aspect_ratio))
recircle = cv2.resize(graycircle, (225,225), interpolation=cv2.INTER_AREA)
bitand = cv2.bitwise_and(graystar, recircle, mask=None)
bitor = cv2.bitwise_or(graystar, recircle, mask=None)
bitxor = cv2.bitwise_xor(graystar, recircle, mask=None)
bitnot = cv2.bitwise_not(recircle, mask=None)
#print(graystar.shape)
#print(recircle.shape)
#cv2.imshow("and", bitand)
#cv2.imshow("or", bitor)
#cv2.imshow("xor", bitxor)
#cv2.imshow("not", bitnot)
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
matrix = np.ones(img.shape, dtype = img.dtype)*50
matrix1 = np.ones(img.shape, dtype = img.dtype)*0.3
matrix2 = np.ones(img.shape, dtype = img.dtype)*2.0

brighter = cv2.add(img, matrix)
darker =  cv2.subtract(img, matrix)

condarker = np.uint8(cv2.multiply(np.float64(img), matrix1))
conbrighter = np.uint8(np.clip(cv2.multiply(np.float64(img), matrix2), 0, 255))
#cv2.imshow("darker", condarker)
#cv2.imshow("brighter", conbrighter)
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Application of masking
remcdo = cv2.resize(mcdo, (360,360), interpolation=cv2.INTER_LINEAR)
recolor = cv2.resize(color, (360,360), interpolation=cv2.INTER_LINEAR)
gmcdo = cv2.cvtColor(remcdo, cv2.COLOR_BGR2GRAY)
retval, threshmcdo = cv2.threshold(gmcdo, 150, 255, cv2.THRESH_BINARY)
mask_mcdo = cv2.bitwise_not(threshmcdo, mask=None)
merged = cv2.bitwise_and(recolor, recolor, mask=threshmcdo)
merged2 = cv2.bitwise_and(remcdo, remcdo, mask=mask_mcdo)
res = cv2.add(merged, merged2)
print(gmcdo.shape)
print(recolor.shape)
cv2.imshow("gray", mcdo)
cv2.imshow("merge", merged)
#cv2.imshow("masked", mask_mcdo)
cv2.imshow("merge2", merged2)
cv2.imshow("result", res)
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#cv2.imshow("frmae", gray2)
cv2.waitKey(0)
cv2.destroyAllWindows()