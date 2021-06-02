#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
from tqdm import tqdm


# Set the path to the images captured by the left and right cameras
pathL = "../../CameraCalibration/0.5/calibrationImages/camL/"
pathR = "../../CameraCalibration/0.5/calibrationImages/camR/"

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((4*11, 3), np.float32)
objp[:, :2] = np.mgrid[0:4, 0:11].T.reshape(-1, 2)

img_ptsL = []
img_ptsR = []
obj_pts = []

# Setup Default SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()
# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255
# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 50     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 10e5   # maxArea may be adjusted to suit for your experiment
# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.8
# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87
# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01
double_count_in_column = True # count the double circles in asymmetrical circular grid along the column


blobDetector = cv2.SimpleBlobDetector_create(blobParams)


for i in tqdm(range(1, len(os.listdir(pathL)))):
    imgL = cv2.imread(pathL+"camL_%d.jpg" % i)
    imgR = cv2.imread(pathR+"camR_%d.jpg" % i)
    imgL_gray = cv2.imread(pathL+"camL_%d.jpg" % i, 0)
    imgR_gray = cv2.imread(pathR+"camR_%d.jpg" % i, 0)

    outputL = imgL.copy()
    outputR = imgR.copy()

    retR, cornersR = cv2.findCirclesGrid(
        outputR, (4, 11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
    retL, cornersL = cv2.findCirclesGrid(
        outputL, (4, 11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
    if retR and retL:
        obj_pts.append(objp)
        #cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        #cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(outputR, (4, 11), cornersR, retR)
        cv2.drawChessboardCorners(outputL, (4, 11), cornersL, retL)
        cv2.imshow('cornersR', outputR)
        cv2.imshow('cornersL', outputL)
        cv2.waitKey(0)

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)


# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(
    mtxL, distL, (wL, hL), 1, (wL, hL))

mean_error = 0
for i in range(len(obj_pts)):
    imgpoints2, _ = cv2.projectPoints(
        obj_pts[i], rvecsL[i], tvecsL[i], mtxL, distL)
    error = cv2.norm(img_ptsL[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(obj_pts)))

# undistort
dst = cv2.undistort(imgL_gray, mtxL, distL, None, new_mtxL)
# crop the image
x, y, w, h = roiL
dst = dst[y:y+h, x:x+w]
cv2.imshow('test', dst)
cv2.waitKey(0)

# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(
    mtxR, distR, (wR, hR), 1, (wR, hR))

print(retR)
mean_error = 0
for i in range(len(obj_pts)):
    imgpoints2, _ = cv2.projectPoints(
        obj_pts[i], rvecsR[i], tvecsR[i], mtxR, distR)
    error = cv2.norm(img_ptsR[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(obj_pts)))
# undistort
dst = cv2.undistort(imgR_gray, mtxR, distR, None, new_mtxR)
# crop the image
x, y, w, h = roiR
dst = dst[y:y+h, x:x+w]
cv2.imshow('test', dst)
cv2.waitKey(0)


flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo = (cv2.TERM_CRITERIA_EPS +
                   cv2.TERM_CRITERIA_MAX_ITER, 500000, 0.00000001)


# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
    obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], criteria_stereo, flags)

print()
rectify_scale = 1
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(
    new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1], Rot, Trns, rectify_scale, (0, 0))
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                              imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                               imgR_gray.shape[::-1], cv2.CV_16SC2)
print(retS)

print("Saving paraeters ......")
cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()


cv2.imshow("Left image before rectification", imgL)
cv2.imshow("Right image before rectification", imgR)

Left_nice = cv2.remap(
    imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
Right_nice = cv2.remap(
    imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

cv2.imshow("Left image after rectification", Left_nice)
cv2.imshow("Right image after rectification", Right_nice)
cv2.waitKey(0)

out = Right_nice.copy()
out[:, :, 0] = Right_nice[:, :, 0]
out[:, :, 1] = Right_nice[:, :, 1]
out[:, :, 2] = Left_nice[:, :, 2]

cv2.imshow("Output image", out)
cv2.waitKey(0)
