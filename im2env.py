import numpy as np
import cv2 as cv
import os
import Cali_Cam as cc
import requests
import pprint

import pdb
import argparse


TEST_IMGS_PATH = 'test_imgs/'
rect_lower = 0.89
small_arc_upper = 0.89
small_arc_lower = 0.84  # 0.83
bone_upper = 0.84  # 0.83
bone_lower = 0.75
big_arc_upper = 0.75
big_arc_lower = 0.65
goal_upper = 0.65
padding = 50


class Obstacle:
    def __init__(self, contour, area, moments, x_scale, y_scale, upper_left, cbx, cby, z, Hc2b=None):
        '''
        cbx -> the x coordinate of the origin of the checkerboard in REAL WORLD COORDINATES(mm) in CAMERA FRAME
        cby -> the y coordinate of the origin of the checkerboard in REAL WORLD COORDINATES(mm) in CAMERA FRAME
        z -> the z coordinate of the origin of the checkerboard in REAL WORLD COORDINATES(mm) in CAMERA FRAME
        '''
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        self.solidity = float(area)/hull_area
        self.rect = cv.minAreaRect(contour)
        self.boundingRect = cv.boundingRect(contour)
        self.moments = moments
        self.cx = int(self.moments['m10']/self.moments['m00'])
        self.cy = int(self.moments['m01']/self.moments['m00'])
        self.x_scale = x_scale
        self.y_scale = y_scale
        # self.cbx = upper_left[0]
        # self.cby = upper_left[1]
        self.cbx = cbx
        self.cby = cby
        self.z = z
        self.inferType()

        # if negative angle, rotate wrist counter clockwise
        self.angle = self.rect[2] if self.rect[2] > - \
            45. else self.rect[2] + 90.  # just trust it

    def inferType(self):
        if self.solidity > rect_lower:
            self.type = 'long rectangle'
        elif self.solidity < goal_upper:
            self.type = 'goal'
        elif self.solidity < big_arc_upper and self.solidity > big_arc_lower:
            self.type = 'big arc'
        elif self.solidity < bone_upper and self.solidity > bone_lower:
            self.type = 'bone'
        elif self.solidity < small_arc_upper and self.solidity > small_arc_lower:
            self.type = 'small arc'
        else:
            self.type = 'WTF IS THIS?!?!?!?!'

    def visualize(self, pic):
        box = cv.boxPoints(self.rect)
        box = np.int0(box)
        cv.drawContours(pic, [box], 0, (0, 0, 255), 2)
        (x, y, w, h) = self.boundingRect
        cv.putText(pic, self.type, (x+w, y+h), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 255), 1, cv.LINE_AA)  
        cv.putText(pic, str(self.angle), (x, y), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 255, 0), 1, cv.LINE_AA)  
        centroid = '(' + str(int(self.cx*self.x_scale+self.cbx)) + ', ' + \
                         str(-int(self.cy*self.y_scale+self.cby)) + \
                  ', ' + str(int(self.z)) + ')'
        cv.putText(pic, centroid, (x+20, y+20), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 0, 0), 1, cv.LINE_AA)


def composeEnv(obstacles, row, col):
    '''
    Compose a json object to be used in optimizer
    all positional measurements are in meters
    '''
    global_dict = dict()
    o_list = []

    obs = obstacles[0]
    x_bound = col*obs.x_scale/1000
    y_bound = row*obs.y_scale/1000

    for o in obstacles:
        count = 0
        (x, y), (w, h), _ = o.rect

        # destination
        if o.type == 'goal':
            destination = {'x': (o.cx*o.x_scale + o.cbx) / 1000,  # x is good
                           'y': -(o.cy*o.y_scale + o.cby) / 1000,  # y is actually bottom left, not top left
                           'width': w*o.x_scale / 1000,
                           'height': h*o.y_scale / 1000}
            global_dict['destination'] = destination
        else:
            # i know cx and cy is a better name but let me test this first
            o_dict = {'type': o.type,
                      'x': (o.cx*o.x_scale + o.cbx) / 1000,
                      'y': -(o.cy*o.y_scale + o.cby) / 1000,
                      'width': w*o.x_scale / 1000,
                      'height': h*o.y_scale / 1000}
            o_list.append(o_dict)

    # bounds
    bounds = {'x': [-x_bound/2, x_bound/2], 'y': [-y_bound/2, y_bound/2], 'rotation': [0, 31.4159265359]}
    global_dict['bounds'] = bounds
    # obstacles
    global_dict['obstacles'] = o_list

    # number of obstacles
    global_dict['n_obstacles'] = len(obstacles)-1

    # ball
    ball = {'radius': 0.01,
            'location': [-x_bound/4, y_bound/2], 'linear_velocity': [0.1, -0.05]}
    global_dict['ball'] = ball

    return global_dict


def c2b(cam_pose, bax_pose):
    '''
    pass in cam_pose and bax pose as vectors: x, y, z
    returns the transformation from camera coord to baxter coordinates
    '''
    cx = np.absolute(cam_pose[0])
    cy = np.absolute(cam_pose[1])
    cz = np.absolute(cam_pose[2])
    bx = np.absolute(bax_pose[0])
    by = np.absolute(bax_pose[1])
    bz = np.absolute(bax_pose[2])

    Hc2b = np.array([[0, 0, 1, (bx-cz)/1000],
                     [-1, 0, 0, (by-cx)/1000],
                     [0, -1, 0, (bz-cy)/1000],
                     [0, 0, 0, 1]])
    print('im here')
    # pdb.set_trace()
    return Hc2b

def CamToBax(camPose):

    test_rot = np.array([[0, 0, 1, 0],
                         [-1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 0, 1]])
    #pdb.set_trace()
    # chris = np.dot(test_rot, camPose)
    # print(chris)
    test_trans = np.array([[1, 0, 0, 0.2520],
                          [0, 1, 0, 0.1133],
                          [0, 0, 1, 0.3866],
                          [0, 0, 0, 1]])

    result = np.dot(test_trans, np.dot(test_rot, camPose))
    print ('theirs')
    #pdb.set_trace()

    return result




def fitRectangles(pic, visualize=False):

    # calibrate camera first
    cam = cc.Cali_Cam()
    target_pts, image_pts = cam.extract_points(pic)
    cb_origin, cb_diagonal, x_scale, y_scale = cam.CB_bounds(image_pts)  # will be used later to mask out the checkerboard

    upper_left = np.array([cb_origin[0], cb_diagonal[1]]).astype(int)
    bottom_right = np.array([cb_diagonal[0], cb_origin[1]]).astype(int)

    H, intrinsic = cam.camera_params(target_pts, image_pts, pic)
    dist_z = H[2, 3]

    # checkerboard origin in camera frame, real world coordinates
    cbx = H[0, 3]
    cby = H[1, 3]

    # checkerboard origin, real world coordinates (mm), camera frame
    print ('checkerboard origin x: ', cbx)
    print ('checkerboard origin y: ', cby)
    print ('checkerboard origin z: ', dist_z)

    # camera1 = np.array([-480, -278, 710, 1])
    # camera2 = np.array([-480+240, -278, 710, 1])
    # H = c2b([cbx, cby, dist_z], [967.537, 519.957, 491.942])

    # vec = np.array([0.967, 0.5199, 0.4919, 1])
    # vec1 = np.array([-0.0897096909, -0.013427036, 0.715, 1])
    # vec1 = np.array([0.970, 0.203, 0.4, 1])
    #vec = np.array([0.076022, ])
    # test_rot = np.array([[0, -1, 0, 0],
    #                      [0, 0, -1, 0],
    #                      [1, 0, 0, 0],
    #                      [0, 0, 0, 1]])
    # chris = np.dot(test_rot, vec1)
    # print(chris)
    # test_trans = np.array([[1, 0, 0, 0.967-chris[0]],
    #                       [0, 1, 0, 0.5199-chris[1]],
    #                       [0, 0, 1,0.4919-chris[2]],
    #                       [0, 0, 0, 1]])
    # test_rot = np.array([[0, 0, 1, 0],
    #                      [-1, 0, 0, 0],
    #                      [0, -1, 0, 0],
    #                      [0, 0, 0, 1]])
    # chris = np.dot(test_rot, vec1)
    # print(chris)
    # test_trans = np.array([[1, 0, 0, 0.967-chris[0]],
    #                       [0, 1, 0, 0.203-chris[1]],
    #                       [0, 0, 1,0.4-chris[2]],
    #                       [0, 0, 0, 1]])
 
    # # # test = np.linalg.inv(test)
    
    # chris2 = np.dot(test_trans, chris)
    # print (chris2)

    # extraction
    gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)

    # bw image, white is region of interest
    t = np.ceil(0.43*256)  # 0.43 might be better for other pics
    # assign 1 to the regions that are above t
    ret, thresh = cv.threshold(gray, t, 1, cv.THRESH_BINARY_INV)

    # remove checkerboard regions
    thresh[0:bottom_right[1]+padding, 0:bottom_right[0]+padding] = 0

    # convert BGR to HSV
    hsv = cv.cvtColor(pic, cv.COLOR_BGR2HSV)
    hsv = cv.normalize(hsv.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # h channel to remove the shadow
    h_ch = hsv[:, :, 0]
    h_mask = (h_ch > 0.1).astype(np.uint8)

    # s channel to remove dirty whiteboard
    s_ch = hsv[:, :, 1]
    s_mask = (s_ch > 0.22).astype(np.uint8)

    mask = (thresh + h_mask + s_mask + s_mask) > 2
    mask = mask.astype(np.uint8)
    thresh = mask*thresh

    # post process thresh
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    thresh = cv.erode(thresh, se, iterations=5)

    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, se)

    # median blur
    thresh = cv.medianBlur(thresh, 15).astype(np.uint8)

    # find contours
    im2, contours, hierarchy = cv.findContours(thresh, 1, 2)

    obstacles = []

    # good contours
    for c in contours:
        area = cv.contourArea(c)
        moments = cv.moments(c)
        if (area > 1000 and area < 150000):
            obstacles.append(
                Obstacle(c, area, moments, x_scale, y_scale, upper_left, cbx, cby, dist_z))

    if visualize:
        for obs in obstacles:
            obs.visualize(pic)
        cv.imshow('thresh', thresh*255)
        cv.imshow("Contours", pic)
        cv.waitKey(0)

    # send into compose env for calculating bounds
    row, col = thresh.shape

    return composeEnv(obstacles, row, col)


if __name__ == "__main__":
    # img_names = os.listdir(TEST_IMGS_PATH)
    # imgs = bb1 = [cv.imread(TEST_IMGS_PATH+name) for name in img_names]

    imgs = cv.imread('div_test.png')
    environment = fitRectangles(imgs, visualize=True)
    pprint.pprint(environment)

    init_x = environment['obstacles'][0]['x']  #test
    init_y = environment['obstacles'][0]['y']  #test
    init = [init_x, init_y,  0.715, 1]

    r = requests.post('http://localhost:5000/getpose', json=environment)
    pprint.pprint(r.json())


    final_x = r.json()['obstacles'][0]['x']  #test
    final_y = r.json()['obstacles'][0]['y']  #test
    final =  [final_x, final_y,  0.715, 1]
    rot_final = r.json()['obstacles'][0]['rotation']

    init = CamToBax(init)
    final = CamToBax(final)

    print("Initial Position of Obstacle 0", init)
    print("Final Position of Obstacle 0",final)

    pdb.set_trace()
    pass
