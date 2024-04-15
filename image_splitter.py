import cv2
import numpy as np
cap = cv2.VideoCapture("/home/ahmed/testapartment.mp4")

while(1):
    ret, frame = cap.read()
    img = resize = cv2.resize(frame, (640, 480))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape[:2]
    img_left = img[:,:img_w//2]
    img_right = img[:, img_w//2:]
    mask = np.zeros_like(img_gray)
    mask[:380, 200:440] = 255
    vis = np.concatenate((img_left, img_right), axis = 1)
    cv2.imshow("mask", mask)
    print("height: ", img_h)
    print("width: ", (img_w//2)//2)
    #cv2.imshow("frame2", img_left)
    #cv2.imshow("frame3", img_right)
    cv2.imshow("merged", vis)
    if(cv2.waitKey(1)&0xFF==27):
        break
cap.release()
cv2.destroyAllWindows()