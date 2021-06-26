# ***********************
# **** Main Programm ****
# ***********************

# Package importation
import numpy as np
import cv2
from openpyxl import Workbook  # Used for writing data into an Excel file
#from sklearn.preprocessing import normalize


# Filtering
kernel = np.ones((3, 3), np.uint8)

Q = [[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.93386066e+02],
 [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -2.34396317e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  7.69477289e+02],
 [ 0.00000000e+00,  0.00000000e+00,  3.47341446e-01, -0.00000000e+00]]



# Create the background subtractor object
# Use the last 700 video frames to build the background
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

max_depth = 2.00  # maximum distance the setup can measure (in m)
min_depth = 0.10  # minimum distance the setup can measure (in m)
depth_map = []


def compute_distance(x, y):
    average = 0
    for u in range(-1, 2):
        for v in range(-1, 2):
            average += disp[y+u, x+v]
    average = average/9

    Distance = 48.847*average**(-1.094)
    return np.around(Distance*0.01, decimals=2)

def compute_distance_m_matrix(x, y):
    average = 0
    for u in range(-1, 2):
        for v in range(-1, 2):
            average += disp[y+u, x+v]
    average = average/9

    Distance = M / (average)
    return np.around(Distance, decimals=2)   

def coords_mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("Distance = %.2f m" % compute_distance_m_matrix( x, y))

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

def create_depth_map():
    depth_map = M / (disp)
    mask_temp = cv2.inRange(depth_map, min_depth, max_depth)
    depth_map = cv2.bitwise_and(depth_map, depth_map, mask=mask_temp)


def track_object(frame):
    # Use every frame to calculate the foreground mask and update
    # the background
    fg_mask = back_sub.apply(frame)
 
    # Close dark gaps in foreground object using closing
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
 
    # Remove salt and pepper noise with a median filter
    fg_mask = cv2.medianBlur(fg_mask, 5) 
     
    # Threshold the image to make it either black or white
    _, fg_mask = cv2.threshold(fg_mask,127,255,cv2.THRESH_BINARY)

    # Find the index of the largest contour and draw bounding box
    fg_mask_bb = fg_mask
    contours, hierarchy = cv2.findContours(fg_mask_bb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]

    # If there are no countours
    if len(areas) < 1:
        return
 
    else:
       # Find the largest moving object in the image
       max_index = np.argmax(areas)
    # Draw the bounding box
    cnt = contours[max_index]
    cv2.drawContours(grayL, cnt, -1, (255,0,0), thickness=2)
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(grayL,(x,y),(x+w,y+h),(0,255,0),3)
    # Draw circle in the center of the bounding box
    x2 = x + int(w/2)
    y2 = y + int(h/2)
    cv2.circle(grayL,(x2,y2),4,(0,255,0),-1)
 
    # Print the centroid coordinates (we'll use the center of the
    # bounding box) on the image
    text = "depth: " + str(compute_distance_m_matrix(x2, y2)) + "m"
    cv2.putText(grayL, text, (x2 - 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def append_ply_array(verts, colors):
    global accumulated_verts
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts_new = np.hstack([verts, colors])
    if accumulated_verts is not None:
        accumulated_verts = np.vstack([accumulated_verts, verts_new])
    else:
        accumulated_verts = verts_new    

def to_3d():    
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    append_ply_array(out_points, out_colors)

    disparity_scaled = (disp - min_disp) / num_disp
    disparity_scaled += abs(np.amin(disparity_scaled))
    disparity_scaled /= np.amax(disparity_scaled)
    disparity_scaled[disparity_scaled < 0] = 0
    return np.array(255 * disparity_scaled, np.uint8) 


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
    '..\\..\\CapturedVideos\\samples\\zMovingSampleLeft640x480.avi')
 #   '..\\..\\CapturedVideos\\samples\\sampleLeft.avi')
CamR = cv2.VideoCapture(
    '..\\..\\CapturedVideos\\samples\\zMovingSampleRight640x480.avi')
  #  '..\\..\\CapturedVideos\\samples\\sampleRight.avi')

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage(
    "./improved_params2.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
M = cv_file.getNode("M").real() #box .343 handstand .32
cv_file.release()

counter = 0

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

        
        #track object
        track_object(grayL)

        

        # Show the result for the Depth_image
        # cv2.imshow('Disparity', disp)
        # cv2.imshow('Closing',closing)
        # cv2.imshow('Color Depth',disp_Color)
        # cv2.imshow('Filtered Color Depth', filt_Color)
        cv2.imshow('Image Result', grayL)


        # Mouse click
        cv2.setMouseCallback("Image Result",
                             coords_mouse_disp, grayL)

        counter += 1

        # End the Programme
        #if counter == 83:
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite('c1.png', grayL)
            print('screenshot taken ' +  str(counter))
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
