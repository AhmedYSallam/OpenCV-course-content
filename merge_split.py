import cv2
import numpy as np

img = cv2.imread("/home/ahmed/IMAGES/board.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (640,480), interpolation= cv2.INTER_LINEAR)
#b,g,r = cv2.split(img)
print(img)
print(img.shape)
print(img.dtype)
#b = b+100
#cv2.imshow("b", b)
#cv2.imshow("g", g)
#cv2.imshow("r", r)
#merge = cv2.merge([r, g, b])
#cv2.imshow("merge", merge)
cv2.imshow('frmae', img)
cv2.waitKey(0)
cv2.destroyAllWindows()