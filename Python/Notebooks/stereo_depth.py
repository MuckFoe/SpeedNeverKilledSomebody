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

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


if __name__ == '__main__':
    # Args handling -> check help parameters to understand
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--calibration_file', type=str, required=True, help='Path to the stereo calibration file')
    parser.add_argument('--is_real_time', type=int, required=True, help='Is it camera stream or video')


    left_source = "nvarguscamerasrc sensor-id=0 sensor-mode=3 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    right_source = "nvarguscamerasrc sensor-id=1 sensor-mode=3 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    args = parser.parse_args()

    # is camera stream or video
    if args.is_real_time:
        cap_left = cv2.VideoCapture(left_source, cv2.CAP_GSTREAMER)
        cap_right = cv2.VideoCapture(right_source, cv2.CAP_GSTREAMER)
    else:
        cap_left = cv2.VideoCapture(left_source)
        cap_right = cv2.VideoCapture(right_source)

    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params

    if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
        print("Can't opened the streams!")
        sys.exit(-9)


    while True:  # Loop until 'q' pressed or stream ends

        _, leftFrame = cap_left.read()
        _, rightFrame = cap_right.read()
        height, width, channel = leftFrame.shape  # We will use the shape for remap
        #height = 480
        #width = 640
        #channel = 3
        
        leftFrame = leftFrame.astype(np.float32)
        rightFrame = rightFrame.astype(np.float32)

        # Undistortion and Rectification part!
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        print('Recitification works')
        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        disparity_image = depth_map(gray_left, gray_right)  # Get the disparity map

        # Show the images
        cv2.imshow('left(R)', leftFrame)
        cv2.imshow('right(R)', rightFrame)
        cv2.imshow('Disparity', disparity_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
            break

    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()