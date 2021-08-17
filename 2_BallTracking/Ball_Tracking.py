
#deque will maintain a list of past coordinates of the ball
#deque helps to append and pop very efficiently
from collections import deque
#imutils is an OpenCV convenience library   
from imutils.video import VideoStream
import imutils
import cv2
import numpy as np
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help = 'path to the video file')
#--buffer is the maximum size of the deque, which keeps a track of our
# i.e a small queue will lead to a shorter tracker whereas a larger queue will create a longer tracker 
ap.add_argument('-b', '--buffer', type = int, default = 75, help = 'max buffer size')       
args = vars(ap.parse_args())    #vars return the __dict__ attribute of the given object 

#define the lower and upper boundaries of the 'green' ball in HSV color. To do so refer 'HSV_color_range.png' and 'color_range_detector.py'
#The x-axis represents Hue in [0,180), the y-axis1 represents Saturation in [0,255], the y-axis2 represents S = 255, while keep V = 255.
#To find a color, usually just look up for the range of H and S, and set v in range(20, 255). 
# To find the color of the ball, we look up for the map, and find the best range: H :[30, 45], S: [50, 255], and V: [20, 255]. So the (30, 100, 20), (25, 255, 25) 
ball_color_low = (25, 38, 20)
ball_color_high = (55, 255, 255)

#tracker keeps a track of the coordinates the ball has travelled
tracker = deque(maxlen = args['buffer'])

#if video is not passed as an argument continue with the stream of webcam
if not args.get('video', False):
    vs = VideoStream(src = 0).start()
else:
    vs = cv2.VideoCapture(args['video'])

win_name = 'Ball Tracker'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#the video loop will continue unless 'esc' key is pressed or the video ends
while True:
    frame = vs.read()
    
    #handle the frame from video or cam stream
    if args.get('video', False):
        frame = frame[1]
    else:
        frame = cv2.flip(frame, 1)

    if frame is None:
        break

    #resize the video frame for higher FPS, blur the frame to reduce high frequency noise
    # frame = imutils.resize(frame, width = 500)
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)  #experiment with parameters
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #we will make a mask with the color ranges that we have defined, this actually localizes the ball in out frame
    #the result of the below line is a mask
    mask = cv2.inRange(hsv, ball_color_low, ball_color_high)

    #we will do a series of erosions and dilations to further remove any small blobs that may be on the mask which is not a part of our ball
    #specifically erosions remove white noise and dilations is used to enlarge the targeted object
    mask = cv2.erode(mask, None, iterations = 4)
    mask = cv2.dilate(mask, None, iterations = 2)

    #find the contours of the mask
    cnts, retrieval = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Initialize the current centre coordinate of the ball as None
    center = None

    #check if contours are present
    if len(cnts) > 0:
        #choose the contour with the maximum area
        c = max(cnts, key = cv2.contourArea)
        #we find the enclosing circle
        ((x,y), radius) = cv2.minEnclosingCircle(c)

        #moments is a feature of contour and we can use it to find the centroid/centre of our masked ball
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        #draw the enclosing circle if it's radius is prominent
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255, 0), 2)
            cv2.circle(frame, center, 5, (0,0,255), -1)
    
    #update the tracker queue
    tracker.appendleft(center)

    #now we just need to show the tracker line
    for i in range(1, len(tracker)):
        #for the tracker, it is important the ball is present in the current frame as well as the previous one
        if tracker[i] and tracker[i-1]:
            thickness = 2
            cv2.line(frame, tracker[i-1], tracker[i], (0,255,255), thickness)
    
    cv2.imshow(win_name, frame)
    
    key = cv2.waitKey(1) & 0xFF
    #break out of the loop when 'esc' key is pressed
    if key == 27:
        break


    #if we are not using a video file then stop the camera stream
if not args.get('video', False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()
