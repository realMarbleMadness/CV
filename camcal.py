# camera calibration from Matlab
# intrinsic matrix
# [924.1445, 0,        0;
#         0  924.4962, 0;
#  636.5713, 364.4404, 1.0000];

# translation vector
# [-26.0926760711823, -75.2907126441399, 356.198491132291;
#  -58.8655561961630, -47.5131397988825, 332.713622165353;
#  110.447768389927,  -82.4766412888175, 380.108397982818;
#  -85.1829032463149, -60.5713540161995, 371.113519889493]

# rotation vector
# [-0.0729499476053123, 0.0539929640991140,  -0.0319396998028909;
#  -0.0266439921984914, -0.0334236523996129, -0.370729220417304;
#  -0.0620577613031699, 0.0177595495278574,  1.55797962170929;
#  -0.0428921283837814, 0.273701857577316,   -0.0689224712801643]

import numpy as np
import cv2
import glob
import pdb

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


img = cv2.imread('cb1.PNG')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (9,9),None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (9,9), corners2,ret)
    cv2.imshow('img',img)
    cv2.waitKey(500)

cv2.destroyAllWindows()



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

pdb.set_trace()