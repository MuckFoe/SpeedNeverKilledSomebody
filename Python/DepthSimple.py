import numpy as np
import cv2
from matplotlib import pyplot as plt

# linkes bild
imgL = cv2.imread('bugshooter_L.jpg', 0)
# rechtes bild
imgR = cv2.imread('bugshooter_R.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()