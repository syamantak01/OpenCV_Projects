import cv2
from collections import deque
import numpy as np
import sys
import time

#A required callback function required in the createTrackbar method
# def nothing(x):
#     pass

# def color_range_detector():
#     s = 0

#     source = cv2.VideoCapture(s)
#     if not source.isOpened():
#         print('Cannot Open Video')
#         exit()

#     win_name = 'Color Range TrackBars'
#     cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#     #A required callback function required in the createTrackbar method

#     #We will find the color range of our pen
#     #For that we will create 6 trackbars that will control the lower and upper range  of Hue, Saturation, Vue
#     cv2.createTrackbar("Lower H", win_name, 0, 179, nothing)
#     cv2.createTrackbar("Lower S", win_name, 0, 255, nothing)
#     cv2.createTrackbar("Lower V", win_name, 0, 255, nothing)
#     cv2.createTrackbar("Upper H", win_name, 179, 179, nothing)
#     cv2.createTrackbar("Upper S", win_name, 255, 255, nothing)
#     cv2.createTrackbar("Upper V", win_name, 255, 255, nothing)

#     while True:
#         has_frame, frame = source.read()
#         if not has_frame:
#             break
        
#         frame = cv2.flip(frame, 1)

#         blurred = cv2.GaussianBlur(frame, (11, 11), 0)
#         #convert BGR to HSV color space
#         hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#         l_h = cv2.getTrackbarPos("Lower H", win_name)
#         l_s = cv2.getTrackbarPos("Lower S", win_name)
#         l_v = cv2.getTrackbarPos("Lower V", win_name)
#         u_h = cv2.getTrackbarPos("Upper H", win_name)
#         u_s = cv2.getTrackbarPos("Upper S", win_name)
#         u_v = cv2.getTrackbarPos("Upper V", win_name)

#         pen_color_low = np.array([l_h, l_s, l_v])
#         pen_color_high = np.array([u_h, u_s, u_v])

#         mask = cv2.inRange(hsv, pen_color_low, pen_color_high)
#         # Perform the morphological operations to get rid of the noise.
#         # Erosion Eats away the white part while dilation expands it.
#         mask = cv2.erode(mask,None,iterations = 2)
#         mask = cv2.dilate(mask,None,iterations = 2)

#         #converting into BGR for stacking
#         col_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

#         #stack all frames to show it
#         stacked = np.hstack((col_mask, frame))
#         cv2.imshow(win_name, cv2.resize(stacked, None, fx = 0.8, fy = 0.8))

#         key = cv2.waitKey(1) & 0xFF
#         #break out of the loop when 'esc' key is pressed
#         if key == 27:
#             break
#     source.release()
#     cv2.destroyAllWindows()

#     lower_range = (l_h, l_s, l_v)
#     upper_range = (u_h, u_s, u_v)
#     return lower_range, upper_range

def write(lower_range, upper_range):
    s = 0
    source = cv2.VideoCapture(s)
    if not source.isOpened():
        print('Cant open Camera')
        time.sleep(2)
        exit()

    win_name = 'Virtual_Board'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    #We’ll be drawing on a black board and then merge that board with the frame. 
    # This is because we are getting a new frame on every iteration so we can’t draw on the actual frame.
    board = None
    x1, y1 = 0, 0

    while True:
        has_frame, frame = source.read()
        if not has_frame:
            break

        frame = cv2.flip(frame, 1)

        #initialise the board as a black image of the same size as that of the frame
        if board is None:
            board = np.zeros_like(frame)

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)  #experiment with parameters
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_range, upper_range)

        #erosions remove white noise and dilations is used to enlarge the targeted object
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)

        #Find the contours
        cnts, retrieval = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #check if at least one contour is present
        if len(cnts) > 0:
            #choose the contour with the largest area
            c = max(cnts, key = cv2.contourArea)

            ((x2,y2), radius) = cv2.minEnclosingCircle(c)

            if radius > 7:
                #when we write for the first time when the pen dissappear while writing, the previous coordinate points will be unsaved
                #if previous points are unsaved, save them with the current coordinate points
                if x1 == 0 and y1 == 0:
                    x1, y1 = x2, y2
                else:
                    #join the current point and the previous point to draw a line
                    board = cv2.line(board, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                #After drawing the line the current point become the previous point
                x1, y1 = x2, y2
            else:
                x1, y1 = 0, 0

        #The next 4 lines are for smoother drawing
        ret , mask = cv2.threshold(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(board, board, mask = mask)
        background = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(mask))
        frame = cv2.add(foreground,background)
    
        # frame = cv2.add(frame, board)
        frame = cv2.putText(frame, 'Press C to clear', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow(win_name, frame)

        # stacked = np.hstack((board, frame))
        # cv2.imshow(win_name, stacked)

        
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('c'):
            board = None

    source.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #uncomment the color range detector to find the color range
    # lower_range, upper_range = color_range_detector()
    write(lower_range = (105, 150, 20), upper_range = (120, 255, 255))

    
