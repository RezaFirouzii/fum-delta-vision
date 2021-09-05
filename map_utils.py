import math
import cv2 as cv
import numpy as np
from numpy.linalg import inv



plate_center = (600, 350)
o1, o2 = plate_center
plate_radius = 300


def draw_holes(img, coords, color=(0, 0, 255)):
    for center in coords:
        x, y = map(int, center)
        cv.circle(img, (x, y), 7, color, 2)

    return img



def get_similarity(frame, coords):
    holes_count = 0
    bound = 15
    for x, y in coords:
        x, y = int(x), int(y)
        # cv.imshow('', cv.rectangle(frame.copy(), (x-bound, y-bound), (x+bound, y+bound), (255, 255, 255), 1))
        # cv.waitKey()
        roi = frame[y-bound: y+bound, x-bound: x+bound]
        if np.count_nonzero(roi):
            holes_count += 1

    return holes_count




def create_map_sample(frame, org, angles):
    map_sample = np.array([])
    similarity = 0
    for theta in angles:
        
        # Drawing the three small circles
        t = [theta, theta-np.pi, theta-(3*np.pi/2)]
        small_circle_offset = 225
        x = small_circle_offset*np.cos(t)+o1
        y = small_circle_offset*np.sin(t)+o2
        coords = list(zip(x, y))

        map_similarity = get_similarity(frame, coords)
        if map_similarity < similarity:
            continue

        # Drawing the base circles
        # img = 255 * np.ones(shape=[720, 1280, 3], dtype=np.uint8)
        img = org.copy()
        cv.circle(img, center=plate_center, radius=plate_radius, color=(200,100,100), thickness=5)
        cv.circle(img, center=plate_center, radius=32, color=(200,100,100), thickness=5)
        img = draw_holes(img, coords, color=(0, 0, 0))

        t = np.linspace((2*np.pi/12)+theta, (2*np.pi/12)+2*np.pi+theta, 7)
        for i in t:
            outer_circle_offset = 42
            x = outer_circle_offset*np.cos(i)+o1
            y = outer_circle_offset*np.sin(i)+o2
            cv.circle(img, center=(int(x), int(y)), radius=5, color=(200,100,100), thickness=2)

        # Drawing the 22 outer circles
        t = np.linspace((2*np.pi/44)+theta, (2*np.pi/44)+2*np.pi+theta, 23)
        outer_circle_offset = 275
        x = outer_circle_offset*np.cos(t)+o1
        y = outer_circle_offset*np.sin(t)+o2
        coords = list(zip(x, y))
        img = draw_holes(img, coords)
        print(np.array(list(zip(x, y))) - np.array(plate_center))

        # Drawing the inner circles
        t = np.linspace((np.pi/2)+theta, (np.pi/2)+2*np.pi+theta, 3)
        outer_circle_offset = 160
        x = outer_circle_offset*np.cos(t)+o1
        y = outer_circle_offset*np.sin(t)+o2
        coords = list(zip(x, y))
        img = draw_holes(img, coords)

        t = [np.pi*73.3/180+theta, np.pi*106.7/180+theta, np.pi*253.3/180+theta, np.pi*286.7/180+theta]
        outer_circle_offset = 104.4
        x = outer_circle_offset*np.cos(t)+o1
        y = outer_circle_offset*np.sin(t)+o2
        coords = list(zip(x, y))
        img = draw_holes(img, coords)

        t = [np.pi*69.44/180+theta, np.pi*110.56/180+theta, np.pi*249.44/180+theta, np.pi*290.56/180+theta]
        outer_circle_offset = 170.88
        x = outer_circle_offset*np.cos(t)+o1
        y = outer_circle_offset*np.sin(t)+o2
        coords = list(zip(x, y))
        img = draw_holes(img, coords)

        t = [np.pi*48.1/180+theta, np.pi*131.99/180+theta, np.pi*228.01/180+theta, np.pi*311.99/180+theta]
        outer_circle_offset = 134.54
        x = outer_circle_offset*np.cos(t)+o1
        y = outer_circle_offset*np.sin(t)+o2
        coords = list(zip(x, y))
        img = draw_holes(img, coords)

        t = [np.pi*53.13/180+theta, np.pi*126.87/180+theta, np.pi*233.13/180+theta, np.pi*306.87/180+theta]
        outer_circle_offset = 200
        x = outer_circle_offset*np.cos(t)+o1
        y = outer_circle_offset*np.sin(t)+o2
        coords = list(zip(x, y))
        img = draw_holes(img, coords)

        t = [np.pi*33.69/180+theta, np.pi*146.31/180+theta, np.pi*213.69/180+theta, np.pi*326.31/180+theta]
        outer_circle_offset = 180.28
        x = outer_circle_offset*np.cos(t)+o1
        y = outer_circle_offset*np.sin(t)+o2
        coords = list(zip(x, y))
        img = draw_holes(img, coords)

        similarity = map_similarity
        map_sample = img

    return map_sample


def get_rotation_angles(src):
    coords = [(825.0, 350.0), (375.0, 350.0), (600.0, 575.0)]
    # Coordination after rotation
    x2, y2 = src

    angles = []
    for x1, y1 in coords:
        try:
            tan1 = (y1 - o2) / (x1 - o1)            
        except ZeroDivisionError:
            tan1 = (y1 - o2) / (x1 - o1 + 1e-16)

        try:
            tan2 = (y2 - o2) / (x2 - o1)
        except ZeroDivisionError:
            tan2 = (y2 - o2) / (x2 - o1 + 1e-16)

        tan = (tan2 - tan1 ) / (1 + tan1*tan2)
        angles.append(math.atan(tan))


    return angles