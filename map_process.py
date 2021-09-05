import math
import pickle
import imageio
import cv2 as cv
import numpy as np
from map_utils import *
import Calibration.calibration as clb


def invhole(x, y):
    a = 25
    x = int(2*O1 - x - a/2)
    y = int(2*O2 - y - a/2)
    hole = np.count_nonzero(frame[y: y+2*a, x: x+2*a])

    return hole


def draw(x, y):
    a = 25
    x = int(2*O1 - x - a/2)
    y = int(2*O2 - y - a/2)

    cv.imshow('i', cv.rectangle(img.copy(), (x, y), (x+2*a, y+2*a), (0, 255, 0), 2))
    cv.imshow('f', frame.copy())
    # cv.imshow('f', cv.rectangle(frame.copy(), (x, y), (x+2*a, y+2*a), (255, 255, 255), 1))
    cv.waitKey()



if __name__ == "__main__":
    # reader = imageio.get_reader(1, cv.CAP_DSHOW)
    # fps = int(reader.get_meta_data()['fps'])
    
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Video was not opened!")

    coeffs = clb.load_coefficients('calibration/')
    matrix = pickle.load(open('calibration/prespective_matrix.pickle', 'rb'))
    arr = [0] * 4
    O1, O2 = 480, 360
    MSE = 0
    frame_count = 0
    while frame_count < 2000:
        res, frame = cap.read()
        if not res:
            break

        frame = cv.resize(frame, None, fx=1.5, fy=1.5)
        rows, cols = frame.shape[:2]

        # distortion
        frame, roi = clb.undistort(frame, coeffs)
        # prespective transform
        frame = cv.warpPerspective(frame, matrix, (cols, rows))

        # img = frame.copy()
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 9)

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
        # frame = cv.medianBlur(frame, 3)

        # contours, hierarchy = cv.findContours(frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        # contours = list(filter(lambda x: 50 < cv.contourArea(x) < 75, contours))
        # contours = list(map(cv.minEnclosingCircle, contours))
        # contours = list(sorted(contours, key=lambda x: x[1]))
        # contours = contours[:min(5, len(contours))]
        # contours = list(filter(lambda cnt: 220 < np.hypot(cnt[0][0] - O1, cnt[0][1] - O2) < 230, contours))
        # contours = contours[:min(3, len(contours))]
        # contours = list(map(lambda x: x[0], contours))

        # if len(contours) == 0:
        #     continue

        # src = contours[0]
        # angles = get_rotation_angles(src)
        # if angles[0] < 0:
        #     angles[0] = 2*math.pi + angles[0]
        # angles[1] = math.pi   + angles[1]
        # angles.append(math.pi + angles[2])

        # map_sample = create_map_sample(frame, img, angles)
        # # cv.imshow('sample', map_sample)
        # # cv.imshow('org', cv.circle(img.copy(), tuple(map(int, src)), 5, (0, 255, 0), 2))    
        # # cv.waitKey()

        # for cnt in contours:
        #     x, y = cnt
        #     e = np.hypot(x - O1, y - O2) - 225
        #     MSE += e**2
        
        # arr[min(3, len(contours))] += 1
        # frame_count += 1

        # break
        # print('----------------')
        cv.imshow('frame', frame)
        # cv.imshow('map', map_sample)
        # cv.imshow('img', img)
        if cv.waitKey(30) == 27:
            break
    
    print(arr)
    SUM = arr[1] + 2*arr[2] + 3*arr[3]
    # print("MSE:", MSE / SUM)
    print("Frames:", frame_count)
    cap.release()
    cv.destroyAllWindows()
