import numpy as np
import cv2
import json

CamL = cv2.VideoCapture(
    'C:\\Users\\Moe\\Desktop\\UniversityCoursesGit\\SpeedNeverKilledSomebody\\CapturedVideos\\samples\\zMovingSampleLeft640x480.avi')
CamR = cv2.VideoCapture(
    'C:\\Users\\Moe\\Desktop\\UniversityCoursesGit\\SpeedNeverKilledSomebody\\CapturedVideos\\samples\\zMovingSampleRight640x480.avi')

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage(
    "./improved_params2.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()


def save_config(v):
    config_map = {}
    config_map['numDisparities'] = numDisparities / 16
    config_map['blocksize'] = (blockSize - 5) / 2
    config_map['preFilterType'] = preFilterType
    config_map['preFilterSize'] = (preFilterSize - 5) / 2
    config_map['preFilterCap'] = preFilterCap
    config_map['textureThreshold'] = textureThreshold
    config_map['uniquenessRatio'] = uniquenessRatio
    config_map['speckleRange'] = speckleRange
    config_map['speckleWindowSize'] = speckleWindowSize / 2
    config_map['disp12MaxDiff'] = disp12MaxDiff
    config_map['minDisparity'] = minDisparity
    with open('config.json', 'w') as fp:
        json.dump(config_map, fp)


def nothing(x):
    pass


cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 400, 400)

cv2.namedWindow('disp2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp2', 600, 600)

cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)
cv2.createTrackbar('saveConfig', 'disp', 1, 1, lambda v: save_config(v))

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

while True:

    # Capturing and storing left and right camera images
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()

    # Proceed only if the frames have been captured
    if retL and retR:

        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        # Applying stereo image rectification on the left image
        Left_nice = cv2.remap(imgL_gray,
                              Left_Stereo_Map_x,
                              Left_Stereo_Map_y,
                              cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT,
                              0)

        # Applying stereo image rectification on the right image
        Right_nice = cv2.remap(imgR_gray,
                               Right_Stereo_Map_x,
                               Right_Stereo_Map_y,
                               cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT,
                               0)

        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice, Right_nice)
        disparity = np.uint8(disparity)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.
        #disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_DEEPGREEN)
        # Converting to float32
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # Displaying the disparity map
        cv2.imshow("disp", disparity)
        cv2.imshow("disp2", Left_nice)

        # Close window using esc key
        if cv2.waitKey(50) == 27:
            break
    else:
        print('no video')
        CamL.set(cv2.CAP_PROP_POS_FRAMES, 0)
        CamR.set(cv2.CAP_PROP_POS_FRAMES, 0)

print()
