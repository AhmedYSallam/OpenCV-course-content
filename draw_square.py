import cv2
import numpy as np

img = cv2.imread("/home/ahmed/board.png")
print(img)
print(img.shape)
print(img.dtype)

#for i in range(0, 200):
#    for j in range (0, 200):
#        img[0+i,0+j] = (0, 255, 0)

img[100:150, 100:150] = 200
cv2.imshow("frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()