import numpy as np
from numpy import linalg as LA
import math

import pdb
import cv2 as cv
import os
import glob


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

row_corner = 10
col_corner = 7
checker_size = 24.  # in mm

objp = np.zeros((row_corner*col_corner, 3), np.float32)

objp = np.zeros ((70, 3), np.float32)
count = 0
for i in range(0, 10):
    for j in range(6, -1, -1):
        objp[count, 0] = i
        objp[count, 1] = j
        count += 1

objp = objp*checker_size

# pdb.set_trace()

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img = cv.imread('all2.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (col_corner, row_corner), None)
# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (col_corner, row_corner), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(0)
cv.destroyAllWindows()

# top left checkerboard anchor
CB_origin = corners2[6]
CB_right = corners2[-1]
CB_down = corners2[0]
CB_diagonal = corners2[(row_corner-1)*col_corner] 

# get intrinsic and extrinsics
ret, intrinsic, dist, rotation, translation = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
retval, rvec, tvec = cv.solvePnP(objpoints[0], imgpoints[0], intrinsic, dist)

# get
rot_mat = cv.Rodrigues(rotation[0])[0]
cam_pos = np.matrix(rot_mat).T*np.matrix(translation[0])
cam_pos[2] -= 1000

print (cam_pos)

pdb.set_trace()








