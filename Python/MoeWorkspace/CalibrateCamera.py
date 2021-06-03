#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm


# Set the path to the images captured by the left and right cameras
pathL = "../../CameraCalibration/0.9/calibrationImages/camL/"
pathR = "../../CameraCalibration/0.9/calibrationImages/camR/"

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in range(1, 107):
    imgL = cv2.imread(pathL+"camL_%d.jpg" % i, 0)
    imgR = cv2.imread(pathR+"camR_%d.jpg" % i, 0)
    imgL_non_gray = cv2.imread(pathL+"camL_%d.jpg" % i)
    imgR_non_gray = cv2.imread(pathR+"camR_%d.jpg" % i)

    # this part here is only for the sanitycheck pictures
    #imgL = cv2.imread(pathL+"left%d.jpg" % i)
    #imgR = cv2.imread(pathR+"right%d.jpg" % i)
    #imgL_gray = cv2.imread(pathL+"left%d.jpg" % i, 0)
    #imgR_gray = cv2.imread(pathR+"right%d.jpg" % i, 0)

    #outputL = imgL.copy()
    #outputR = imgR.copy()

    retR, cornersR = cv2.findChessboardCorners(imgR, (9, 6), None)
    retL, cornersL = cv2.findChessboardCorners(imgL, (9, 6), None)

    if (True == retR) & (True == retL):
        obj_pts.append(objp)
        cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria)
        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)


# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,
                                                        img_ptsL,
                                                        imgL.shape[::-1], None, None)
hL, wL = imgL.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL,
                                               distL,
                                               (wL, hL), 1, (wL, hL))
print('retL')
print(retL)

# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,
                                                        img_ptsR,
                                                        imgR.shape[::-1], None, None)
hR, wR = imgR.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR,
                                               distR,
                                               (wR, hR), 1, (wR, hR))


print('retR')
print(retR)

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo = (cv2.TERM_CRITERIA_EPS +
                   cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
test_Q = None
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
                                                                                    img_ptsL,
                                                                                    img_ptsR,
                                                                                    mtxL,
                                                                                    distL,
                                                                                    mtxR,
                                                                                    distR,
                                                                                    imgR.shape[::-1],
                                                                                    criteria_stereo,
                                                                                    flags)
print(retS)

rectify_scale = 1  # if 0 image croped, if 1 image nor croped
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL,
                                                                          distL,
                                                                          new_mtxR,
                                                                          distR,
                                                                          imgL.shape[::-1],
                                                                          Rot,
                                                                          Trns,
                                                                          rectify_scale, (0, 0))
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                              imgL.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                               imgR.shape[::-1], cv2.CV_16SC2)

print("Saving parameters ......")
cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()


cv2.imshow("Left image before rectification", imgL)
cv2.imshow("Right image before rectification", imgR)

Left_nice = cv2.remap(
    imgL_non_gray, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
Right_nice = cv2.remap(
    imgR_non_gray, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

cv2.imshow("Left image after rectification", Left_nice)
cv2.imshow("Right image after rectification", Right_nice)
cv2.waitKey(0)

out = Right_nice.copy()
out[:, :, 0] = Right_nice[:, :, 0]
out[:, :, 1] = Right_nice[:, :, 1]
out[:, :, 2] = Left_nice[:, :, 2]


# Setting parameters for StereoSGBM algorithm
minDisparity = 0
numDisparities = 64
blockSize = 10
disp12MaxDiff = 1
uniquenessRatio = 20
speckleWindowSize = 10
speckleRange = 8

# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                               numDisparities=numDisparities,
                               blockSize=blockSize,
                               disp12MaxDiff=disp12MaxDiff,
                               uniquenessRatio=uniquenessRatio,
                               speckleWindowSize=speckleWindowSize,
                               speckleRange=speckleRange
                               )

# Calculating disparith using the StereoSGBM algorithm
disp = stereo.compute(Left_nice, Right_nice).astype(np.float32)
disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)

# Displaying the disparity map
cv2.imshow("disparity", disp)

#pointCloud = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=0)

# print(pointCloud)
#np.save("./pointCloud", pointCloud)
#cv2.imshow("Output image", out)
cv2.waitKey(0)
