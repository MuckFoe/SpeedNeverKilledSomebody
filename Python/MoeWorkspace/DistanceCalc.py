# ***********************
# **** Main Programm ****
# ***********************

# Package importation
import numpy as np
import cv2
from openpyxl import Workbook  # Used for writing data into an Excel file
from sklearn.preprocessing import normalize

# Filtering
kernel = np.ones((3, 3), np.uint8)


def coords_mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print x,y,disp[y,x],filteredImg[y,x]
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y+u, x+v]
        average = average/9
        print('average: = ' + str(average))
        #Distance = -0.0000007*average**(3) +0.0003* average**(2) - 0.0343*average + + 1.8291
        # 48,847x-1,094
        Distance = 0.34 / average
        Distance = np.around(Distance, decimals=2)
        print('Distance: ' + str(Distance)+' m')

# This section has to be uncommented if you want to take mesurements and store them in the excel
# ws.append([counterdist, average])
# print('Measure at '+str(counterdist)+' cm, the dispasrity is ' + str(average))
# if (counterdist <= 85):
# counterdist += 3
# elif(counterdist <= 120):
# counterdist += 5
# else:
# counterdist += 10
# print('Next distance to measure: '+str(counterdist)+'cm')


# Mouseclick callback
wb = Workbook()
ws = wb.active

# *******************************************
# ***** Parameters for the StereoVision *****
# *******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=5,
                               P1=8*3*window_size**2,
                               P2=32*3*window_size**2)

# Used for the filtered image
# Create another stereo for right this time
stereoR = cv2.ximgproc.createRightMatcher(stereo)

# WLS FILTER Parameters
lmbda = 80000
sigma = 2
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# *************************************
# ***** Starting the StereoVision *****
# *************************************

# Call the two cameras
# CamR= cv2.VideoCapture(0)   # Wenn 0 then Right Cam and wenn 2 Left Cam
# CamL= cv2.VideoCapture(2)

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


while True:
    # Start Reading Camera images
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()
    if retL and retR:
        # Rectify the images on rotation and alignement
        # Rectify the image using the kalibration parameters founds during the initialisation
        Left_nice = cv2.remap(
            frameL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        Right_nice = cv2.remap(
            frameR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Draw Red lines
    # for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
    # Left_nice[line*20,:]= (0,0,255)
    # Right_nice[line*20,:]= (0,0,255)
    ##
    # for line in range(0, int(frameR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
    # frameL[line*20,:]= (0,255,0)
    # frameR[line*20,:]= (0,255,0)

        # Show the Undistorted images
        # cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
        # cv2.imshow('Normal', np.hstack([frameL, frameR]))

        # Convert from color(BGR) to gray
        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

        # Compute the 2 images for the Depth_image
        disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
        dispL = disp
        dispR = stereoR.compute(grayR, grayL)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)

        # Using the WLS filter
        filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
        filteredImg = cv2.normalize(
            src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)
        # cv2.imshow('Disparity Map', filteredImg)
        # Calculation allowing us to have 0 for the most distant object able to detect
        disp = ((disp.astype(np.float32) / 16)-min_disp)/num_disp

    # Resize the image for faster executions
    # dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

        # Filtering the Results with a closing filter
        # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)
        closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)

        # Colors map
        dispc = (closing-closing.min())*255
        # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
        dispC = dispc.astype(np.uint8)
        # Change the Color of the Picture into an Ocean Color_Map
        disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)
        filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

        # Show the result for the Depth_image
        # cv2.imshow('Disparity', disp)
        # cv2.imshow('Closing',closing)
        # cv2.imshow('Color Depth',disp_Color)
        cv2.imshow('Filtered Color Depth', filt_Color)

        # Mouse click
        cv2.setMouseCallback("Filtered Color Depth",
                             coords_mouse_disp, filt_Color)

        # End the Programme
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    else:
        print('no video')
        CamL.set(cv2.CAP_PROP_POS_FRAMES, 0)
        CamR.set(cv2.CAP_PROP_POS_FRAMES, 0)
# Save excel
# wb.save("data4.xlsx")

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
