import numpy as np
import cv2 as cv
import pdb


# read image
# all = cv.imread('all_parts.PNG')
# all = cv.imread('Bshort_curve.PNG')
# all = cv.imread('Glong_curve.PNG')
# all = cv.imread('Goal.PNG')
# all = cv.imread('Gshort_curve.PNG')
# all = cv.imread('Gstraight.PNG')
# all = cv.imread('Olong_curve.PNG')
# all = cv.imread('Pstraight.PNG')
bb1 = cv.imread('realb1.png')
bb2 = cv.imread('realb2.png')
bb3 = cv.imread('realb3.png')

pic = bb3

gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)

# bw image, white is region of interest
t = np.ceil(0.42*256)
ret, thresh = cv.threshold(gray, t, 1, cv.THRESH_BINARY_INV)  # assign 1 to the regions that are above t

# convert BGR to HSV
hsv = cv.cvtColor(pic, cv.COLOR_BGR2HSV)
hsv = cv.normalize(hsv.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
# for some reason openCV is 1.4010989010989023 smaller than matlab in h
# for some reason openCV is 1.0211670480549206 smaller than matlab in s, maybe not
h_const = 1.4010989010989023
s_const = 1.0211670480549206

# h channel to remove the shadow
h_ch = hsv[:,:,0]
h_mask = (h_ch > 0.1).astype(np.uint8)

# s channel to remove dirty whiteboard
s_ch = hsv[:,:,1]
s_mask = (s_ch > 0.22).astype(np.uint8)

mask = (thresh + h_mask + s_mask + s_mask) > 2
mask = mask.astype(np.uint8)
thresh = mask*thresh

# post process thresh
se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2, 2))
thresh = cv.erode(thresh, se, iterations = 2)
se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(12, 12))
thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, se)

# cv.imshow("Contours", thresh*255);
# cv.waitKey(0)
# pdb.set_trace()

# find contours
im2, contours, hierarchy = cv.findContours(thresh, 1, 2)

contours_filtered = []
solidity = []

for c in contours:
    area = cv.contourArea(c)
    if (area > 1000 and area < 150000):
        contours_filtered.append(c)

        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        solidity.append(float(area)/hull_area)


for i, (cont, solid)  in enumerate(zip(contours_filtered, solidity)):
    (x,y,w,h) = cv.boundingRect(cont)

    # fit tight rectangle
    rect = cv.minAreaRect(cont)
    angle = rect[2]
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(pic,[box],0,(0,0,255),2) 

    if solid > 0.9:
        cv.putText(pic, 'long rectangle',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
        cv.putText(pic, str(angle),(x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)  # red text
    elif solid < 0.65:
        cv.putText(pic,'goal',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
        cv.putText(pic, str(angle),(x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)  # red text
    elif solid < 0.75 and solid > 0.65:
        cv.putText(pic,'big arc',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
        cv.putText(pic, str(angle),(x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)  # red text
    elif solid < 0.83 and solid > 0.75:
        cv.putText(pic,'dildo',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
        cv.putText(pic, str(angle),(x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)  # red text
    elif solid < 0.9 and solid > 0.83:
        cv.putText(pic,'small arc',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
        cv.putText(pic, str(angle),(x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)  # red text  
    else:
        cv.putText(pic,'WTF IS THIS?!?!?!?!',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text          




print (solidity)

cv.imshow('thresh', thresh)
cv.imshow("Contours",pic);
cv.waitKey(0)




# # blob detector
# detector = cv.SimpleBlobDetector_create(params)
# keypoints = detector.detect(thresh)