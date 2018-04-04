import numpy as np
import cv2 as cv
import pdb


# read image
all = cv.imread('all_parts.PNG')
# all = cv.imread('Bshort_curve.PNG')
# all = cv.imread('Glong_curve.PNG')
# all = cv.imread('Goal.PNG')
# all = cv.imread('Gshort_curve.PNG')
# all = cv.imread('Gstraight.PNG')
# all = cv.imread('Olong_curve.PNG')
# all = cv.imread('Pstraight.PNG')

gray_all = cv.cvtColor(all, cv.COLOR_BGR2GRAY)

# bw image, white is region of interest
ret, thresh = cv.threshold(gray_all, 127, 255, cv.THRESH_BINARY_INV)  

# convert BGR to HSV
hsv = cv.cvtColor(all, cv.COLOR_BGR2HSV)
hsv = cv.normalize(hsv.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
h_ch = hsv[:,:,0]
mask = h_ch > 0.1
mask.astype(np.uint8)

thresh = mask*thresh

# post process thresh
se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2, 2))
thresh = cv.erode(thresh, se, iterations = 2)

se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(12, 12))
thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, se)

# find contours
im2, contours, hierarchy = cv.findContours(thresh, 1, 2)

contours_filtered = []
solidity = []

for c in contours:
    area = cv.contourArea(c)
    if (area > 500 and area < 150000):
        contours_filtered.append(c)

        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        solidity.append(float(area)/hull_area)


for i, (cont, solid)  in enumerate(zip(contours_filtered, solidity)):
    (x,y,w,h) = cv.boundingRect(cont)
    if solid > 0.9:
        cv.putText(all, 'long rectangle',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
    elif solid < 0.65:
        cv.putText(all,'goal',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
    elif solid < 0.75 and solid > 0.65:
        cv.putText(all,'big arc',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
    elif solid < 0.9 and solid > 0.75:
        cv.putText(all,'small arc',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text  
    else:
        cv.putText(all,'WTF IS THIS?!?!?!?!',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text          

    cv.rectangle(all,(x,y),(x+w,y+h),(255,0,0),2)


print (solidity)

cv.imshow('thresh', thresh)
cv.imshow("Contours",all);
cv.waitKey(0)




# # blob detector
# detector = cv.SimpleBlobDetector_create(params)
# keypoints = detector.detect(thresh)