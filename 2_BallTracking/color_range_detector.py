import cv2
from collections import deque
import numpy as np
import sys


s = 0

def nothing(x):
    pass

source = cv2.VideoCapture(s)
if not source.isOpened():
    print('Cannot Open Video')
    exit()

win_name = 'Color Range TrackBars'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#A required callback function required in the createTrackbar method

#We will find the color range of our pen
#For that we will create 6 trackbars that will control the lower and upper range  of Hue, Saturation, Vue
cv2.createTrackbar("Lower H", win_name, 0, 179, nothing)
cv2.createTrackbar("Lower S", win_name, 0, 255, nothing)
cv2.createTrackbar("Lower V", win_name, 0, 255, nothing)
cv2.createTrackbar("Upper H", win_name, 179, 179, nothing)
cv2.createTrackbar("Upper S", win_name, 255, 255, nothing)
cv2.createTrackbar("Upper V", win_name, 255, 255, nothing)

while True:
    has_frame, frame = source.read()
    if not has_frame:
        break
    
    frame = cv2.flip(frame, 1)

    #convert BGR to HSV color space
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor( blurred, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("Lower H", win_name)
    l_s = cv2.getTrackbarPos("Lower S", win_name)
    l_v = cv2.getTrackbarPos("Lower V", win_name)
    u_h = cv2.getTrackbarPos("Upper H", win_name)
    u_s = cv2.getTrackbarPos("Upper S", win_name)
    u_v = cv2.getTrackbarPos("Upper V", win_name)

    pen_color_low = np.array([l_h, l_s, l_v])
    pen_color_high = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, pen_color_low, pen_color_high)
    # Perform the morphological operations to get rid of the noise.
    # Erosion Eats away the white part while dilation expands it.
    mask = cv2.erode(mask,None,iterations = 3)
    mask = cv2.dilate(mask,None,iterations = 2)

    col_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    stacked = np.hstack((col_mask, frame))

    cv2.imshow(win_name, cv2.resize(stacked, None, fx = 0.8, fy = 0.8))

    key = cv2.waitKey(1) & 0xFF
    #break out of the loop when 'esc' key is pressed
    if key == 27:
        break
source.release()
cv2.destroyAllWindows()


