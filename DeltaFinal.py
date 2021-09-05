import cv2
import imageio
import numpy as np
import cv2 as cv
from numpy.linalg import inv
import math
from time import time


# import matplotlib.pyplot as plt
# from IPython.display import display
# from PIL import Image

M = np.array([[0.86293147, 1.17246123, -130.32453727], [1.0130065, -1.06928464, 424.45597799]])
M = np.vstack([M, [0, 0, 1]])
# print(M)

print("[INFO] starting video stream...")


# video_capture = cv2.VideoCapture('C:\Users\FrsCo\PycharmProjects\pythonProject\webcam2.mp4') # We turn the webcam on.
# while True: # We repeat infinitely (until break):
#    _, frame = video_capture.read() # We get the last frame.
#    cv2.imshow('Video', frame) # We display the outputs.
#    cv2.waitKey(100)
#    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
#        break # We stop the loop.

# video_capture.release() # We turn the webcam off.
# cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.

def detect(frame, vel):
    t0 = time()
    image = frame
    # rows,cols,ch = image.shape
    # image = cv2.warpAffine(image,M,(cols * 2 ,rows * 2))
    im_bw = image[:, :, 1]
    blur = cv2.GaussianBlur(im_bw, (11, 11), 0)
    im_bw = cv2.Canny(blur, 50, 50)
    # display(Image.fromarray(im_bw))
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0,255,0), 3)
    im_bw2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh2 = cv.threshold(im_bw2, 200, 255, cv.THRESH_BINARY)
    contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    X = np.array([])
    Y = np.array([])
    R = np.array([])
    clactime=0


   # for i in range(len(contours2)):
    #    cnt2 = contours2[i]
     #   area2 = cv.contourArea(cnt2)
      #  perimeter2 = cv.arcLength(cnt2, True)
       # if 400 > perimeter2 > 300:
        #    (q, e), radius1 = cv.minEnclosingCircle(cnt2)
         #   print("YES")

    for i in range(len(contours)):
        cnt = contours[i]
        # epsilon = 0.2*cv.arcLength(cnt,True)
        # cnt = cv.approxPolyDP(cnt,epsilon,True)
        area = cv.contourArea(cnt)
        # print(area)
        perimeter = cv.arcLength(cnt, True)
        # print(perimeter)
        # cv2.drawContours(image, contours, i, (255,0,0), 3)
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        # cv.circle(image,center,radius,(0,255,0),2)
        # cv.circle(image,center,1,(0,255,0),2)
        # Delta=area//perimeter
        # DeltatoBe=radius//2
        # if Delta=DeltatoBe:
        # cv.circle(image,center,radius,(0,255,0),2)

        # if 170<area<800:
        #   if 50<perimeter<120:
        #     #print(area)
        #     #print(perimeter)
        #     cv.circle(image,center,radius,(0,255,0),2)
        #     cv.circle(image,center,1,(0,255,0),2)
        #     cv2.putText(image, "center" + str(center), center , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)
        #     #cv2.putText(image, "" + str(center) , center , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 100, 255), 1)
        #     continue
        if 600<x and x<810:
            if 190<y and y<380:
                continue
        elif y<50:
            continue
        elif x<250:
            continue

        if x < 727:
            if 0.832*(x-352) > y:
                continue
        elif x >= 727:
            if 312-(0.489*(x-727)) > y:
                continue


        if 40<area<170:
          if 20<perimeter<100:
            #print(area)
            #print(perimeter)
            X = np.append(X, x)
            Y = np.append(Y, y)
            cv.circle(image, center, radius, (255, 0, 0), 2)
            cv.circle(image, center, 1, (255, 0, 0), 2)
            # cv2.putText(image, "center" + str(center), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(image, "" + str((coor_prime[0], coor_prime[1])), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 80, 100, 255), 1, cv2.LINE_AA)
        #print(area)
        # if 170<area:
        #   if perimeter < 100:
        #     # print(area)

        if radius < 20 and radius > 10:
            if area > 60:
                X = np.append(X, x)
                Y = np.append(Y, y)
                cv.circle(image, center, radius, (0, 255, 0), 2)
                cv.circle(image, center, 1, (0, 255, 0), 2)
                #cv2.putText(image, "center" + str(center), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0, 255), 1,cv2.LINE_AA)
            elif area > 10:
                X = np.append(X, x)
                Y = np.append(Y, y)
                cv.circle(image, center, radius, (0, 0, 255), 2)
                cv.circle(image, center, 1, (0, 0, 255), 2)
                #cv2.putText(image, "center" + str(center), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0, 255), 1,cv2.LINE_AA)
    t1 = time()
    calctime=t1-t0
    calctime = calctime + 1e-10
    Time.append(calctime)

        # if area<200:
        # if area>10:
        #  cv.circle(image,center,radius,(255,0,0),2)
        # cv.circle(image,center,1,(255,0,0),2)
    # display(Image.fromarray(image))
    # cv2.imwrite('img.jpg',image)
    # x_min = np.min(X)
    y_max = np.max(Y)
    L = np.argmax(Y)
    center2 = (int(X[L]), int(y_max))
    cv.circle(image, center2, 30, (0, 0, 255), 2)
    cv.circle(image, center2, 1, (0, 0, 255), 2)
    cv2.putText(image, "center" + str(center2), center2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 0), 1)
    cv2.putText(image, "Velocity = " + "{0:.5}".format(float(vel)) + " Rad/S", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255, 0), 1)
    cv2.putText(image, "Calculation Time = " +"{0:.3}".format(1/calctime) + " fps", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0, 0),1)
    print(1/calctime)
    frame = image

    coor = np.array([X[L], y_max, 1])
    coor_tr = np.matmul(inv(M), coor)
    return frame, coor_tr[0], coor_tr[1]


reader = imageio.get_reader('delta.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('webcam7.mp4', fps=fps)
print(fps)

vel = 0
h = 1
VEL=[]
Time = []
for i, frame in enumerate(reader):
    frame, x, y_max = detect(frame, vel)
    writer.append_data(frame)
    #print(i)
    # Velocity Determination
    if h == 1:
        x_old = x
        y_max_old = y_max
        h = h + 1
    else:
        # Calculating the tan(theta) in each step
        x_new = x
        y_max_new = y_max
        tan_old = (y_max_old - 400) / (x_old - 400)
        tan_new = (y_max_new - 400) / (x_new - 400)
        # Calculating the tan(theta_a-theta_b) in each step
        tan = (tan_new - tan_old) / (1 + tan_new * tan_old)
        # Calculating the rotation angle (theta)
        ra = math.atan(tan)
        vel = ra / (1 / fps)
        if vel<0:
            try:
                vel=VEL[i-2]
            except IndexError:
                vel = 0
        #print(vel)
        VEL.append(vel)

        x_old = x_new
        y_max_old = y_max_new

# cv2.imwrite('img.jpg',frame)
writer.close()

# np.savetxt("VEL.csv", VEL, delimiter=",")
# np.savetxt("Time.csv", Time, delimiter=",")