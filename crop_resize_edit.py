import cv2
import numpy as np

dim = (480,480)
img = cv2.imread("/home/ahmed/home.jpeg")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cropped = img[119:239, 119:239]
res = cv2.resize(img, dim, interpolation= cv2.INTER_LINEAR)
flipped = cv2.flip(img, 1)
print(img)
print(img.shape)
print(img.dtype)
cv2.imshow("frame", img)
cv2.imshow("Cropped", cropped)
cv2.imshow("resize", res)
cv2.imshow("flip", flipped)
cv2.waitKey(0)
cv2.destroyAllWindows()