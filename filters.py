import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
feature_params = dict(maxCorners = 500,
                      qualityLevel = 0.2,
                      minDistance = 15,
                      blockSize = 9)
preview = 0
canny = 1
feature = 2
blur = 3
choice = 0
while(1):
    ret, frame = cap.read()
    img = frame.copy()

    if(choice == preview):
        result = img
    elif(choice == canny):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(result, 120,151)
    elif(choice == blur):
        result = cv2.blur(img, (13,13))
    elif(choice == feature):
        result = img
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, **feature_params)
        if( corners is not None ):
            for x, y in np.float32(corners).reshape(-1,2):
                cv2.circle(result, (x,y), 4, (0, 255, 0), -1)
        #for i in corners:
           #  x,y = i.ravel()
           #  cv2.circle(result,(x,y),10,(0,255,0),1)
    
    cv2.imshow("frame", result)
    k = cv2.waitKey(1)
    if(k==27):
        break
    elif (k==ord('1')):
        choice = canny
    elif(k==ord('2')):
        choice = feature
    elif(k==ord('3')):
        choice = blur
    elif(k==ord('4')):
        choice = preview

cap.release()
cv2.destroyAllWindows()