import numpy as np
import pdb
import cv2 as cv
import os

cap = cv.VideoCapture('marble.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

count = 0 
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if count == 0:
        prev_frame = frame
        count += 1
        continue

    #background subtraction
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    sub = gray-prev_gray




    cv.imshow("frame", sub)
    cv.waitKey(0)
    # pdb.set_trace()

    prev_frame = frame
    count += 1