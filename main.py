import pickle
import cv2 as cv
import numpy as np
import Calibration.calibration as clb


if __name__ == "__main__":

    cap = cv.VideoCapture(1, cv.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Video was not opened!")

    coeffs = clb.load_coefficients('calibration/')
    matrix = pickle.load(open('calibration/prespective_matrix.pickle', 'rb'))
    
    while True:
        res, frame = cap.read()
        if not res:
            break
        # frame = cv.resize(frame, None, fx=1.5, fy=1.5)
        rows, cols = frame.shape[:2]
        # frame = cv.resize(frame, (1280, 720))
        cv.imwrite('samples/frame.jpg', frame)
        break
        # distortion
        # frame, roi = clb.undistort(frame, coeffs)
        # prespective transform
        # frame = cv.warpPerspective(frame, matrix, (cols, rows))

        # frame_copy = frame.copy()
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 9)

        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # frame  = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

        # frame = cv.medianBlur(frame, 3)

        cv.imshow("Chg", frame)
        # cv.imshow("org", frame_copy)
        if cv.waitKey(30) == 27:
            break

    cap.release()
    cv.destroyAllWindows()