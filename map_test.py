import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math


def draw_holes(a, b, color=(0, 0, 255)):
    coords = zip(a, b)
    for center in coords:
        x, y = map(int, center)
        cv.circle(img, (x, y), 7, color, 2)


theta = 0
theta = theta*np.pi/180
plt.figure(num=1, figsize=[6,6])

# Drawing the base circles
img = 255 * np.ones(shape=[720, 960, 3], dtype=np.uint8)
plate_center = (480, 360)
o1, o2 = plate_center
plate_radius = 300
cv2.circle(img, center=plate_center, radius=plate_radius, color=(200,100,100), thickness=5)
cv2.circle(img, center=plate_center, radius=32, color=(200,100,100), thickness=5)
t = np.linspace((2*np.pi/12)+theta, (2*np.pi/12)+2*np.pi+theta, 7)
for i in t:
  outer_circle_offset = 42
  a = outer_circle_offset*np.cos(i)+o1
  b = outer_circle_offset*np.sin(i)+o2
  cv2.circle(img, center=(int(a),int(b)), radius=5, color=(200,100,100), thickness=2)


# Drawing the three small circles
t1 = [theta, theta-np.pi, theta-(3*np.pi/2)]
small_circle_offset = 225
a1 = small_circle_offset*np.cos(t1)+o1
b1 = small_circle_offset*np.sin(t1)+o2
plt.imshow(img),plt.scatter(a1, b1, s=32)
draw_holes(a1, b1, color=(0, 0, 0))
print(list(zip(a1, b1)))

# Drawing the 22 outer circles
t2 = np.linspace((2*np.pi/44)+theta, (2*np.pi/44)+2*np.pi+theta, 23)
outer_circle_offset = 275
a2 = outer_circle_offset*np.cos(t2)+o1
b2 = outer_circle_offset*np.sin(t2)+o2
plt.imshow(img),plt.scatter(a2, b2, s=90, c ="white", edgecolor ="red")
draw_holes(a2, b2)

# Drawing the inner circles
t3 = np.linspace((np.pi/2)+theta, (np.pi/2)+2*np.pi+theta, 3)
outer_circle_offset = 160
a3 = outer_circle_offset*np.cos(t3)+o1
b3 = outer_circle_offset*np.sin(t3)+o2
plt.imshow(img),plt.scatter(a3, b3, s=90, c ="white", edgecolor ="red")
draw_holes(a3, b3)

t4 = [np.pi*73.3/180+theta, np.pi*106.7/180+theta, np.pi*253.3/180+theta, np.pi*286.7/180+theta]
outer_circle_offset = 104.4
a4 = outer_circle_offset*np.cos(t4)+o1
b4 = outer_circle_offset*np.sin(t4)+o2
plt.imshow(img),plt.scatter(a4, b4, s=90, c ="white", edgecolor ="red")
draw_holes(a4, b4)

t5 = [np.pi*69.44/180+theta, np.pi*110.56/180+theta, np.pi*249.44/180+theta, np.pi*290.56/180+theta]
outer_circle_offset = 170.88
a5 = outer_circle_offset*np.cos(t5)+o1
b5 = outer_circle_offset*np.sin(t5)+o2
plt.imshow(img),plt.scatter(a5, b5, s=90, c ="white", edgecolor ="red")
draw_holes(a5, b5)

t6 = [np.pi*48.1/180+theta, np.pi*131.99/180+theta, np.pi*228.01/180+theta, np.pi*311.99/180+theta]
outer_circle_offset = 134.54
a6 = outer_circle_offset*np.cos(t6)+o1
b6 = outer_circle_offset*np.sin(t6)+o2
plt.imshow(img),plt.scatter(a6, b6, s=90, c ="white", edgecolor ="red")
draw_holes(a6, b6)

t7 = [np.pi*53.13/180+theta, np.pi*126.87/180+theta, np.pi*233.13/180+theta, np.pi*306.87/180+theta]
outer_circle_offset = 200
a7 = outer_circle_offset*np.cos(t7)+o1
b7 = outer_circle_offset*np.sin(t7)+o2
plt.imshow(img),plt.scatter(a7, b7, s=90, c ="white", edgecolor ="red")
draw_holes(a7, b7)

t8 = [np.pi*33.69/180+theta, np.pi*146.31/180+theta, np.pi*213.69/180+theta, np.pi*326.31/180+theta]
outer_circle_offset = 180.28
a8 = outer_circle_offset*np.cos(t8)+o1
b8 = outer_circle_offset*np.sin(t8)+o2
plt.imshow(img),plt.scatter(a8, b8, s=90, c ="white", edgecolor ="red")
draw_holes(a8, b8)

# cv.imwrite('samples/map.jpg', img)
cv.imshow('', img)
cv.waitKey()
cv.destroyAllWindows()