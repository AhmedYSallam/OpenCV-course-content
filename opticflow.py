import cv2
import numpy as np
import time
import math

#cap = cv2.VideoCapture("/home/ahmed/VIDEOS/11.mp4")
cap = cv2.VideoCapture(0)

param_features = dict(maxCorners = 10,
                      qualityLevel = 0.8,
                      minDistance = 7, 
                      blockSize = 5)
param_lk = dict(winSize = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

trajectories = []
trajectoryLength = 5
update_rate = 1
index_frame = 0
mag = 0


def first(img_gray, prev_gray, trajectories, sumRight, sumLeft, img, mag):
    prev_img, new_img = prev_gray, img_gray
    p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
    p1, status, error = cv2.calcOpticalFlowPyrLK(prev_img, new_img, p0, None, **param_lk)
    p2, status, error = cv2.calcOpticalFlowPyrLK(new_img, prev_img, p1, None, **param_lk)
    d = abs(p0-p2).reshape(-1, 2).max(-1)
    good = d<1
    trajectories_new = []
    i=0
    #print("p1: ", p1)
    #print("p0: ", p0)
    for trajectory, (x,y), flag in zip(trajectories, p1.reshape(-1,2), good):
            if(x>200 and x<440 and y>100 and y<480):
                mag= math.sqrt(math.pow(p1[i][0][0]-p0[i][0][0],2)+math.pow(p1[i][0][1]-p0[i][0][1],2))
            if not flag:
                continue
            trajectory.append((x,y))
            if(len(trajectory)>trajectoryLength):
                del trajectory[0]
            trajectories_new.append(trajectory)
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            if(x>320):
                sumRight += mag
            else:
                sumLeft += mag
            i+=1
        #foe = calc_foe(p0, p1)
        #print(foe)
        #cv2.circle(img, (int(foe[0]),int(foe[1])) , 8, (255,0,255), -1)
    trajectories = trajectories_new
    cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
    cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
    return


def second(img_gray, sumRight, sumLeft):
    mask = np.zeros_like(img_gray)
    mask[100:, 200:440] = 255

        # Lastest point in latest trajectory
    for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
        cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
    p = cv2.goodFeaturesToTrack(img_gray, mask = mask, **param_features)
    if p is not None:
            # If good features can be tracked - add that to the trajectories
        for x, y in np.float32(p).reshape(-1, 2):
            trajectories.append([(x, y)])
    
    print("right: ", sumRight)
    print("left", sumLeft)
    print("abso:", abs(sumLeft-sumRight))
    if(sumLeft>sumRight and (abs(sumLeft-sumRight)>190)):
        print("move right")
       
    elif(sumLeft<sumRight and (abs(sumLeft-sumRight)>190)):
        print("move left")

    else:
        print("move_forward")
    return mask

if __name__ == '__main__':
    while(1):
        #timer = time.time()
        ret, frame = cap.read()
        img = cv2.resize(frame, (640, 480))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_gray = cv2.GaussianBlur(img_gray_wb, (21,21), 0)
        #img_gray = cv2.bilateralFilter(img_gray_wb,11,180,220)
        sumRight = 0
        sumLeft = 0
        threshold = 15

        if(len(trajectories)>0):
            time.sleep(0.01)
            first(img_gray, prev_gray, trajectories, sumRight, sumLeft, img, mag) 

        if index_frame % update_rate == 0:
            time.sleep(0.01)
            mask = second(img_gray, sumRight, sumLeft)

        #end = time.time()
        #fps = 1 / (end-timer)    
        #cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        index_frame += 1
        prev_gray = img_gray
        #cv2.imshow("blur", img_gray)
        cv2.imshow("frame2", img)
        cv2.imshow("mask", mask)
        k = cv2.waitKey(1)
        if(k & 0xFF == 27):
            break
    cap.release()
    cv2.destroyAllWindows()