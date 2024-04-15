import cv2
import numpy as np
#cv2.CAP_V4L2
cap = cv2.VideoCapture(1)

while(1):
    ret, frame = cap.read()
    img = frame.copy()
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    cv2.imshow("frame", img1)
    print(img1.shape)
    if(cv2.waitKey(1)&0xFF==27):
        break
cap.release()
cv2.destroyAllWindows()