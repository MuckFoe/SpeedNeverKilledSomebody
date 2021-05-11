# ------------------------------------------------------------
# CALCULATE DISPARITY (DEPTH MAP)
# Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
# and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

# StereoSGBM Parameter explanations:
# https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

import cv2 as cv
import numpy as np
import sys


# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 11
min_disp = -64
max_disp = 160
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 20
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 200
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
disp12MaxDiff = 0

stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)



def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        preFilterCap=63,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1
    visual_multiplier = 6

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg

cap_left = cv.VideoCapture(
    'C:\\Users\\Moe\\Desktop\\UniversityCoursesGit\\SpeedNeverKilledSomebody\\CapturedVideos\\sampleLeft.avi')
cap_right = cv.VideoCapture(
    'C:\\Users\\Moe\\Desktop\\UniversityCoursesGit\\SpeedNeverKilledSomebody\\CapturedVideos\\sampleRight.avi')

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

    disparity_SGBM = stereo.compute(leftFrame, rightFrame)

    # Normalize the values to a range from 0..255 for a grayscale image
    disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                  beta=0, norm_type=cv.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)

    dept_map = depth_map(leftFrame, rightFrame)  # Get the disparity map

    # Show the images
    cv.imshow('Disparity', dept_map)
    #cv.imshow("Disparity", disparity_SGBM)

    if cv.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
        break

# Release the sources.
cap_left.release()
cap_right.release()
cv.destroyAllWindows()
