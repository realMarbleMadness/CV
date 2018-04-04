import numpy as np
import cv2 as cv
import glob
import pdb

cap = cv.VideoCapture('calibrate.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        gray_all = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # bw image, white is region of interest
        # ret, thresh = cv.adaptiveThreshold(gray_all, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)  
        thresh = cv.adaptiveThreshold(gray_all, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 71, 2)  

        # convert BGR to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        hsv = cv.normalize(hsv.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        h_ch = hsv[:,:,0]
        mask = h_ch > 0.12
        mask.astype(np.uint8)

        thresh = mask*thresh

        # post process thresh
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2, 2))
        thresh = cv.erode(thresh, se, iterations = 2)

        se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15, 15))
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, se)

        # flood fill test
        im_floodfill = thresh.copy()
        h, w = thresh.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv.floodFill(im_floodfill, mask, (0,0), 255);
 
        # Invert floodfilled image
        im_floodfill_inv = cv.bitwise_not(im_floodfill)
         
        # Combine the two images to get the foreground.
        thresh = thresh | im_floodfill_inv


        # find contours
        im2, contours, hierarchy = cv.findContours(thresh, 1, 2)

        contours_filtered = []
        solidity = []

        for c in contours:
            area = cv.contourArea(c)
            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            s = float(area)/(hull_area+1)

            if (area > 2000 and area < 40000 and s > 0.5):
                contours_filtered.append(c)
                solidity.append(s)

                # print (area)

        # pdb.set_trace()

        for i, (cont, solid)  in enumerate(zip(contours_filtered, solidity)):
            (x,y,w,h) = cv.boundingRect(cont)
            if solid > 0.9:
                cv.putText(frame, 'long rectangle',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
            elif solid < 0.65:
                cv.putText(frame,'goal',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
            elif solid < 0.75 and solid > 0.65:
                cv.putText(frame,'big arc',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text
            elif solid < 0.9 and solid > 0.75:
                cv.putText(frame,'small arc',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text  
            else:
                cv.putText(frame,'WTF IS THIS?!?!?!?!',(x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)  # red text          

            # fit tight rectangle
            rect = cv.minAreaRect(cont)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame,[box],0,(0,0,255),2) 

        cv.imshow('Frame', frame)
        # cv.waitKey(0)

        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else: 
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()