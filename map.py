import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
theta = 300
theta = theta*np.pi/180
plt.figure(num=1, figsize=[6,6])

# Drawing the base circles
img = 255 * np.ones(shape=[800, 800, 3], dtype=np.uint8)
plate_center = 400
plate_radius = 300
cv2.circle(img, center=(plate_center,plate_center), radius=plate_radius, color=(200,100,100), thickness=5)
cv2.circle(img, center=(plate_center,plate_center), radius=32, color=(200,100,100), thickness=5)
t = np.linspace((2*np.pi/12)+theta, (2*np.pi/12)+2*np.pi+theta, 7)
for i in t:
  outer_circle_offset = 42
  a = outer_circle_offset*np.cos(i)+plate_center
  b = outer_circle_offset*np.sin(i)+plate_center
  cv2.circle(img, center=(int(a),int(b)), radius=5, color=(200,100,100), thickness=2)


# Drawing the three small circles
t1 = [theta, theta-np.pi, theta-(3*np.pi/2)]
small_circle_offset = 225
a1 = small_circle_offset*np.cos(t1)+plate_center
b1 = small_circle_offset*np.sin(t1)+plate_center
plt.imshow(img),plt.scatter(a1, b1, s=32)

# Drawing the 22 outer circles
t2 = np.linspace((2*np.pi/44)+theta, (2*np.pi/44)+2*np.pi+theta, 23)
outer_circle_offset = 275
a2 = outer_circle_offset*np.cos(t2)+plate_center
b2 = outer_circle_offset*np.sin(t2)+plate_center
plt.imshow(img),plt.scatter(a2, b2, s=90, c ="white", edgecolor ="red")

# Drawing the inner circles
t3 = np.linspace((np.pi/2)+theta, (np.pi/2)+2*np.pi+theta, 3)
outer_circle_offset = 160
a3 = outer_circle_offset*np.cos(t3)+plate_center
b3 = outer_circle_offset*np.sin(t3)+plate_center
plt.imshow(img),plt.scatter(a3, b3, s=90, c ="white", edgecolor ="red")

t4 = [np.pi*73.3/180+theta, np.pi*106.7/180+theta, np.pi*253.3/180+theta, np.pi*286.7/180+theta]
outer_circle_offset = 104.4
a4 = outer_circle_offset*np.cos(t4)+plate_center
b4 = outer_circle_offset*np.sin(t4)+plate_center
plt.imshow(img),plt.scatter(a4, b4, s=90, c ="white", edgecolor ="red")

t5 = [np.pi*69.44/180+theta, np.pi*110.56/180+theta, np.pi*249.44/180+theta, np.pi*290.56/180+theta]
outer_circle_offset = 170.88
a5 = outer_circle_offset*np.cos(t5)+plate_center
b5 = outer_circle_offset*np.sin(t5)+plate_center
plt.imshow(img),plt.scatter(a5, b5, s=90, c ="white", edgecolor ="red")

t6 = [np.pi*48.1/180+theta, np.pi*131.99/180+theta, np.pi*228.01/180+theta, np.pi*311.99/180+theta]
outer_circle_offset = 134.54
a6 = outer_circle_offset*np.cos(t6)+plate_center
b6 = outer_circle_offset*np.sin(t6)+plate_center
plt.imshow(img),plt.scatter(a6, b6, s=90, c ="white", edgecolor ="red")

t7 = [np.pi*53.13/180+theta, np.pi*126.87/180+theta, np.pi*233.13/180+theta, np.pi*306.87/180+theta]
outer_circle_offset = 200
a7 = outer_circle_offset*np.cos(t7)+plate_center
b7 = outer_circle_offset*np.sin(t7)+plate_center
plt.imshow(img),plt.scatter(a7, b7, s=90, c ="white", edgecolor ="red")

t8 = [np.pi*33.69/180+theta, np.pi*146.31/180+theta, np.pi*213.69/180+theta, np.pi*326.31/180+theta]
outer_circle_offset = 180.28
a8 = outer_circle_offset*np.cos(t8)+plate_center
b8 = outer_circle_offset*np.sin(t8)+plate_center
plt.imshow(img),plt.scatter(a8, b8, s=90, c ="white", edgecolor ="red")

# Ploting
# cv2.imwrite('map.jpg', img)
xb = 672.2008965172565
yb = 439.1365805251534
plt.scatter(xb, yb, s=20, c ="white", edgecolor ="green")
plt.imshow(img)
plt.show()

### Calculating the Rotation Angle (theta) ###
# Coordination berofe rotation
xb = 672.2008965172565
yb = 439.1365805251534
# Coordination after rotation
xa = a2[0]
ya = b2[0]
# Calculating the tan(theta) in each step
tan_b = (yb-400)/(xb-400)
tan_a = (ya-400)/(xa-400)
# Calculating the tan(theta_a-theta_b) in each step
tan = (tan_a-tan_b)/(1+tan_a*tan_b)
# Calculating the rotation angle (theta)
ra = math.atan(tan)

print(math.degrees(ra))
print(math.degrees(theta))
# print(theta)
