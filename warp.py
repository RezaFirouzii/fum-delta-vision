import math
import imageio
import cv2 as cv
import numpy as np
import transformer

def fix_rotation(img):
    img_copy = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows, cols = img.shape
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 9)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    img = cv.medianBlur(img, 3)

    contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    roi = max(contours, key=cv.contourArea)
    
    x, y, w, h = cv.boundingRect(roi)
    corners = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]
    src = np.float32(corners)
    # src = np.reshape(src, (len(src), 1, 2))

    # perimeter = cv.arcLength(src, True)
    # corners = cv.approxPolyDP(src, perimeter // 10, True)
    # corners = np.vstack(corners)
    dst = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    matrix = cv.getPerspectiveTransform(src, dst)
    rotated_img = cv.warpPerspective(img_copy, matrix, (cols, rows))
    cv.imshow('', rotated_img)

D1 = 105
D2 = 175
D3 = 275

if __name__ == "__main__":

    cap = cv.VideoCapture('samples/delta.mp4')
    if not cap.isOpened():
        raise IOError("Video was not opened!")

    mse = 0
    count = 0

    reader = imageio.get_reader('samples/delta.mp4')
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer('samples/result.mp4', fps=fps)

    while True:
        res, frame = cap.read()
        if not res:
            break

        mean_error = 0
        holes_count = 0

        img = frame.copy()
        cv.imshow('dfa', img)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_copy = frame.copy()
        # frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 9)

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # frame  = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

        # frame = cv.medianBlur(frame, 3)
        # contours, hierarchy = cv.findContours(frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        # roi = max(contours, key=cv.contourArea)
        # x, y, w, h = cv.boundingRect(roi)
        x, y, w, h = 115, 0, 445, 360
        img = img[y: y+h, x: x+w]
        img = transformer.rotate_along_axis(img, theta=40)
        frame_copy = frame_copy[y: y+h, x: x+w]
        frame_copy = transformer.rotate_along_axis(frame_copy, theta=40)
        # cv.imshow('', frame_copy)
        # cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv.drawContours(frame_copy, roi, -1, (0, 0, 255), 2)

        # res, mask = cv.threshold(frame_copy, 0, 255, cv.THRESH_BINARY)
        # frame_copy = cv.bitwise_and(frame_copy, frame_copy, mask=mask)
        # corners = cv.goodFeaturesToTrack(frame_copy, 1000, 0.0001, 1)
        # corners = list(sorted(corners, key=lambda x: x[0][1]))
        # print(corners[-1], corners[-2])
        # print()

        # corners = np.array([[38, 293], [407, 293]])
        # for item in corners:
        #     # x, y = map(int, item.ravel())
        #     x, y = item
        #     cv.circle(img, (x, y), 5, (0, 0, 255), -1)

        src = np.float32([[0, 0], [w, 0], [38, 293], [407, 293]])
        dst = np.float32([[0, 0], [w, 0], [30, h], [w - 30, h]])
        matrix = cv.getPerspectiveTransform(src, dst)
        img = cv.warpPerspective(img, matrix, (w, h))
        cv.imshow('', img)

        img_copy = img.copy()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 9)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        img = cv.medianBlur(img, 3)

        origin = (w // 2 + 4, h // 2 + 2)
        o1, o2 = origin
        r = w // 2 + 1

        ORIGIN = (0, 0)
        R = 300 # mm

        contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contours = list(filter(lambda x: 50 < cv.contourArea(x) < 175, contours))
        
        factor = 0.1
        smooth_contours = []
        for i in range(len(contours)):
            epsilon = factor * cv.arcLength(contours[i], True)
            approx = cv.approxPolyDP(contours[i], epsilon, True)
            
            x, y, width, height = cv.boundingRect(approx)
            area = width*height
            if len(approx) == 4 and 75 < area < 200:
                smooth_contours.append(contours[i])

                center, radius = cv.minEnclosingCircle(approx)
                radius = int(radius)
                center = tuple(map(int, center))

                x, y = center
                X = ((x - o1) * R) / r
                Y = ((y - o2) * R) / r

                X, Y = round(X, 2), round(Y, 2)

                cv.circle(img_copy, center, radius, (0, 255, 0), 2)
                cv.putText(img_copy, str((X, Y)), center, cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255, 255), 1, cv.LINE_AA)

                e1, e2, e3 = map(lambda d: abs(math.hypot(X, Y) - d), [D1, D2, D3])
                error = min(e1, e2, e3)
                if error < 10:
                    mean_error += error ** 2
                    holes_count += 1

        cv.circle(img_copy, origin, 4, (0, 0, 255), -1)
        # cv.line(img_copy, origin, (origin[0], origin[1]), (255, 0, 255), 2)

        mean_error /= holes_count
        mse += mean_error
        count += 1
        cv.imshow("Final", img_copy)
        writer.append_data(img_copy)
        # cv.imshow("Chg", img)
        if cv.waitKey(30) == 27:
            break

    print("E:", mse / count, "N:", count)
    writer.close()
    cap.release()
    cv.destroyAllWindows()