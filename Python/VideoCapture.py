# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image
#%matplotlib inline

import cv2
from IPython.display import clear_output, Image, display, HTML
import time
#import matplotlib.pyplot as plt
import IPython
import numpy as np
#from matplotlib import pyplot as plt
import base64

from PIL import Image
from io import StringIO
import imutils


left_source = "nvarguscamerasrc sensor-id=0 sensor-mode=3 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
right_source = "nvarguscamerasrc sensor-id=1 sensor-mode=3 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
capL = cv2.VideoCapture(left_source, cv2.CAP_GSTREAMER)
if not capL.isOpened():
    print("Unable to open camera")

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
capR = cv2.VideoCapture(right_source, cv2.CAP_GSTREAMER)
if not capR.isOpened():
    print("Unable to open camera")
    
outL = cv2.VideoWriter('sampleLeft', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640, 480)) 
outR = cv2.VideoWriter('sampleRight', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640, 480))  

stop_time = time.time() + 10

while time.time() < stop_time:
    #clear_output(wait=True)

    # Capture frame-by-frame
    #ret0, frame0 = cap.read()
    #ret1, frame1 = cap2.read()

    # if frame is read correctly ret is True
    #if not ret0:
        #print("Can't receive frame camera 1(stream end?). Exiting ...")
        #break
    #if not ret1:
        #print("Can't receive frame from camera 2 (stream end?). Exiting ...")
        #break
    # Our operations on the frame come here

    #gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    #gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #print('do we see this')

    retL, frameL = capL.read()  # retL is bool, frameL is numpy.ndarray
    retR, frameR = capR.read() 
    
    if retL and retR:
        outL.write(frameL)
        outR.write(frameR)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    else:
        break
    

    #grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    #grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    
    #image_with_edges_L = cv2.Canny(grayL , 50, 500)
    #image_with_edges_R = cv2.Canny(grayR , 50, 500)
    
    #im_rgb = cv2.cvtColor(image_with_edges, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    #imgL = Image.fromarray(image_with_edges_L, 'L')
    #imgR = Image.fromarray(image_with_edges_R, 'L')
    
    # Just for showing the videocap
    #imgL = Image.fromarray(frameL, 'RGB')
    #imgR = Image.fromarray(frameR, 'RGB')

    

# When everything done, release the capture
capL.release()
capR.release()
cv2.destroyAllWindows()
