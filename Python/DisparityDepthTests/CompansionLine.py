import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Read both images and convert to grayscale
img1 = cv.imread('C:\\Users\\Moe\\Desktop\\UniversityCoursesGit\\SpeedNeverKilledSomebody\\CameraCalibration\\calibrationImages\\cam1\\cam1_1.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('C:\\Users\\Moe\\Desktop\\UniversityCoursesGit\\SpeedNeverKilledSomebody\\CameraCalibration\\calibrationImages\\cam2\\cam2_1.jpg', cv.IMREAD_GRAYSCALE)

# ------------------------------------------------------------
# PREPROCESSING

# Compare unprocessed images
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1, cmap="gray")
axes[1].imshow(img2, cmap="gray")
axes[0].axhline(250)
axes[1].axhline(250)
axes[0].axhline(450)
axes[1].axhline(450)
plt.suptitle("Original images")
plt.show()