import math
import pickle
import imageio
import cv2 as cv
import numpy as np
import transformer
import Calibration.calibration as clb



D1 = 105
D2 = 175
D3 = 275


if __name__ == "__main__":
    cap = cv.VideoCapture('samples/test2.mp4')
    if not cap.isOpened():
        raise IOError("Video was not opened!")

    coeffs = clb.load_coefficients('Calibration/')
    mse = 0
    count = 0

    while True:
        res, frame = cap.read()
        if not res:
            break

        mean_error = 0
        holes_count = 0
        
        hight, width = frame.shape[:2]
        frame = frame[:, :width - 200]
        frame = transformer.rotate_along_axis(frame, theta=-15)
        img = frame.copy()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        res, mask = cv.threshold(frame, 100, 255, cv.THRESH_BINARY)
        frame = cv.bitwise_and(frame, frame, mask=mask)
        
        frame = cv.Canny(frame, 245, 255)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        frame  = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)

        contours, hierarchy = cv.findContours(frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        roi = max(contours, key=lambda x: cv.arcLength(x, False))
        x, y, w, h = cv.boundingRect(roi)
        
        img = img[y: y+h, x: x+w]
        layer = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        layer = cv.adaptiveThreshold(layer, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 13)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
        layer = cv.erode(layer, kernel)

        org = (w // 2 + 18, h // 2 + 5)
        o1, o2 = org
        r = 385

        ORIGIN = (0, 0)
        R = 300 # mm

        contours, hierarchy = cv.findContours(layer, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = list(filter(lambda x: 50 < cv.contourArea(x) < 400, contours))
        # print(len(contours))

        # cv.drawContours(img, contours, -1, (0, 255, 0), 2)
        for cnt in contours:
            center, radius = cv.minEnclosingCircle(cnt)
            if radius < 15:
                radius = int(radius)
                center = tuple(map(int, center))
                
                x, y = center
                X = ((x - o1) * R) / r
                Y = ((y - o2) * R) / r

                X, Y = round(X, 2), round(Y, 2)

                cv.circle(img, center, radius, (0, 255, 0), 2)
                cv.putText(img, str((X, Y)), center, cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255, 255), 1, cv.LINE_AA)

                e1, e2, e3 = map(lambda d: abs(math.hypot(X, Y) - d), [D1, D2, D3])
                error = min(e1, e2, e3)
                if error < 10:
                    mean_error += error ** 2
                    holes_count += 1

        mean_error /= holes_count
        mse += mean_error
        count += 1
        
        cv.circle(img, org, 5, (0, 0, 255), -1)
        cv.imshow('frame', frame)
        cv.imshow('layer', layer)
        cv.imshow('final', img)
        if cv.waitKey(30) == 27:
            break

    print("E:", mse / count, "N:", count)
    cap.release()
    cv.destroyAllWindows()