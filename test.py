import os
import pickle
import cv2 as cv
import numpy as np
import Calibration.calibration as clb

# if __name__ == "__main__":
#     path = 'calibration/assets'
#     imgs = os.listdir(path)
#     imgs = list(map(lambda img: os.path.join(path, img), imgs))
#     imgs = list(map(cv.imread, imgs))
    
#     for img in imgs:
#         coeffs = clb.load_coefficients('calibration/')
#         dst, roi = clb.undistort(img, coeffs)

#         img = cv.resize(img, None, fx=.5, fy=.5)
#         roi = cv.resize(roi, None, fx=.5, fy=.5)
#         dst = cv.resize(dst, None, fx=.5, fy=.5)

#         cv.imshow('befor', img)
#         cv.imshow('after', roi)
#         cv.imshow('final', dst)
#         if cv.waitKey(0) == 27:
#               break


if __name__ == "__main__":
    img = cv.imread('samples/frame1.jpg')
    # img = cv.resize(img, None, fx=1.5, fy=1.5)
    rows, cols = img.shape[:2]
    # print(rows, cols)

    src = np.float32([[223, 305], [482, 66], [763, 310], [491, 561]])
    dst = np.float32([[255, 360], [479, 85], [705, 360], [479, 585]])

    mat = cv.getPerspectiveTransform(src, dst)
    img = cv.warpPerspective(img, mat, (cols, rows))

    pickle.dump(mat, open('calibration/prespective_matrix.pickle', 'wb'))
    # mat = pickle.load(open('calibration/prespective_matrix.pickle', 'rb'))
    img = cv.warpPerspective(img, mat, (cols, rows))

    cv.imwrite('hello1.jpg', img)
    cv.imshow('', img)
    cv.waitKey()