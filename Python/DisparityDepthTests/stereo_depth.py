import numpy as np
import cv2
import argparse
import sys


def load_stereo_coefficients(path):
    """ Loads stereo matrix coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]


left_source = "nvarguscamerasrc sensor-id=0 sensor-mode=3 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
right_source = "nvarguscamerasrc sensor-id=1 sensor-mode=3 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

calibration_file = 'stereo.yml'
is_real_time = False

# is camera stream or video
if is_real_time:
    cap_left = cv2.VideoCapture(left_source)
    cap_right = cv2.VideoCapture(right_source)
else:
    cap_left = cv2.VideoCapture(
        'C:\\Users\\Moe\\Desktop\\UniversityCoursesGit\\SpeedNeverKilledSomebody\\CapturedVideos\\sampleLeft.avi')
    cap_right = cv2.VideoCapture(
        'C:\\Users\\Moe\\Desktop\\UniversityCoursesGit\\SpeedNeverKilledSomebody\\CapturedVideos\\sampleRight.avi')

K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(calibration_file)  # Get cams params

if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
    print("Can't opened the streams!")
    sys.exit(-9)

while True:  # Loop until 'q' pressed or stream ends
    # Grab&retreive for sync images
    if not (cap_left.grab() and cap_right.grab()):
        print("No more frames")
        break

    _, leftFrame = cap_left.retrieve()
    _, rightFrame = cap_right.retrieve()
    height, width, channel = leftFrame.shape  # We will use the shape for remap

    stereoMatcher = cv2.StereoBM_create()
    stereoMatcher.setMinDisparity(4)
    stereoMatcher.setNumDisparities(128)
    stereoMatcher.setBlockSize(21)
    stereoMatcher.setSpeckleRange(16)
    stereoMatcher.setSpeckleWindowSize(45)

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
    right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    grayLeft = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)

    cv2.imshow('left(R)', leftFrame)
    cv2.imshow('right(R)', rightFrame)
    cv2.imshow('depth', depth)

# Release the sources.
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
