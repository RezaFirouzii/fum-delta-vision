# imports
import os
import pickle
import cv2 as cv
import numpy as np


def calibrate_chessboard(dir_path, square_size, size):
    width, height = size

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0: width, 0: height].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    imgs = os.listdir(dir_path)
    imgs = list(map(lambda img: os.path.join(dir_path, img), imgs))
    imgs = list(map(cv.imread, imgs))

    for img in imgs:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # finding chessboard corners
        res, corners = cv.findChessboardCorners(gray_img, size, None)
        
        # if found any, add ojbpts and imgpts
        if res:
            objpoints.append(objp)
            corners = cv.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # draw and display the corners
            cv.drawChessboardCorners(img, size, corners, res)
            cv.imshow('', img)
            if cv.waitKey(0) == 27:
                break
    # print(len(objpoints))
    # print(len(imgpoints))

    # calibrate camera
    ret, mtx, dist, rvec, tvec = cv.calibrateCamera(objpoints, imgpoints, gray_img.shape, None, None)

    # projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvec[i], tvec[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(objpoints) 
    return (ret, mtx, dist, rvec, tvec, mean_error)



def save_coefficients(coeffs):
    out = open('camera_coeffs.pickle', 'wb')
    pickle.dump(coeffs, out)
    out.close()


def load_coefficients(path=''):
    inp = open(path + 'camera_coeffs.pickle', 'rb')
    return pickle.load(inp)


def undistort(img, coeffs):
    w, h = img.shape[:2]
    new_mtx, roi = cv.getOptimalNewCameraMatrix(coeffs['mtx'], coeffs['dist'], (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, coeffs['mtx'], coeffs['dist'], None, new_mtx)
    # crop the image
    x, y, w, h = roi
    roi = dst[y:y+h, x:x+w]
    
    return dst, roi



PATH = 'assets'
SIZE = (9, 6)


if __name__ == "__main__":
    ret, mtx, dist, rvec, tvec, err = calibrate_chessboard(PATH, 2.5, SIZE)
    coeffs = {
        "ret": ret,
        "mtx": mtx,
        "dist": dist,
        "rvec": rvec,
        "tvec": tvec,
        "err": err
    }
    save_coefficients(coeffs)
    print(coeffs)