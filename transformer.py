import cv2 as cv
import numpy as np

def rotate_along_axis(img, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
    
    height, width = img.shape[:2]
    # get radius of rotation along 3 axis
    rtheta, rphi, rgamma = get_rad(theta, phi, gamma)
    
    # get ideal focal length on z axis
    # NOTE: Change this section to other axis if needed
    d = np.hypot(height, width)
    focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    dz = focal

    # get projection matrix
    mat = get_M(img, focal, rtheta, rphi, rgamma, dx, dy, dz)
    
    return cv.warpPerspective(img.copy(), mat, (width, height))


def get_M(img, focal, theta, phi, gamma, dx, dy, dz):
    
    h, w = img.shape[:2]
    f = focal

    # Projection 2D -> 3D matrix
    A1 = np.array([ [1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 1],
                    [0, 0, 1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
    
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([ [f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))


def get_rad(theta, phi, gamma):
    return np.deg2rad(theta), np.deg2rad(phi), np.deg2rad(gamma)
