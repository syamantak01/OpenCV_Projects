#!/usr/bin/python

import cv2
import sys

s = 0   #s denotes the source of the video; 0 denotes the continuous stream of video from the webcam

if len(sys.argv)>1:
    s = sys.argv[1]     #Command line arguments can be used to specify the source of the video in the form of filenames

source = cv2.VideoCapture(s)
if not source.isOpened():
    print('Cannot Open Camera')
    exit()

win_name = 'Face Detection Camera'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)    #Creating a window

#returns an instance of the network and we can use that object to perform inference on our test images
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                               "res10_300x300_ssd_iter_140000_fp16.caffemodel")
                                        
#Model Preprocessing Parameters : it's important that we're aware of these because any images that 
# we pass through the model to perform inference on also need to be processed in the same way that 
# the training images were processed 
#(Refer 'https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml' for model parameters)
inp_width = 300
inp_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7    #user-set parameter that determines the sensitivity of detections

while cv2.waitKey(1) != 27:  #ASCII value of 'esc' key is 27
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]


    #Preprocessing on the image frame: Create a 4D Blob(Blob is analogous to tensor)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (inp_width, inp_height), mean, swapRB = False, crop = False)   #swapRB = False because both Caffe and open-cv use the same convention of color channels

    #Run the model
    net.setInput(blob)
    detections = net.forward()  #makes a forward pass through the network and returns blob for first output of specified layer.

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    t, _ = net.getPerfProfile()     #Returns overall time for inference and timings (in ticks) for layers. 
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())   #t is in miliseconds
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)



    

