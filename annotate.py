import cv2
import numpy as np

text = "VIVA LA MONKE REVOLUTION"
font = cv2.FONT_HERSHEY_PLAIN
scale = 1.3

img = cv2.imread("/home/ahmed/lemonk.jpg")
copy = img.copy()
cv2.line(copy, (129, 100), (229,100), (0,255,255), thickness=5, lineType=cv2.LINE_AA)
cv2.circle(copy, (179,179), 50, (255,0,255), thickness=5, lineType=cv2.LINE_AA)
cv2.rectangle(copy, (129,129), (229,229), (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.putText(copy, text,(0, 100), font, scale, (0,100,255), thickness=2, lineType=cv2.LINE_AA)
print(copy.shape)
print(copy.dtype)
cv2.imshow("frame", copy)
cv2.waitKey(0)
cv2.destroyAllWindows()