import numpy as np
import cv2 as cv
import os
import Cali_Cam as cc
import requests
import pprint
import pdb


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
    def __init__(self, contour, area, moments, x_scale, y_scale, upper_left, z):
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
        self.cbx = upper_left[0]
        self.cby = upper_left[1]
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
        centroid = '(' + str(int((self.cx-self.cbx)*self.x_scale)) + ', ' + \
            str(int((self.cy-self.cby)*self.y_scale)) + \
                     ', ' + str(int(self.z)) + ')'
        cv.putText(pic, centroid, (x+20, y+20), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 0, 0), 1, cv.LINE_AA)  


def composeEnv(obstacles, row, col):
    # Compose a json object to be used in optimizer
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
            destination = {'x': (o.cx - o.cbx)*o.x_scale / 1000,  # x is good
                           'y': y_bound-(o.cy - o.cby)*o.y_scale / 1000,  # y is actually bottom left, not top left
                           'width': w*o.x_scale / 1000,
                           'height': h*o.y_scale / 1000}
            global_dict['destination'] = destination
        else:
            # i know cx and cy is a better name but let me test this first
            o_dict = {'type': o.type,
                      'x': (o.cx - o.cbx)*o.x_scale / 1000,
                      'y': (o.cy - o.cby)*o.y_scale / 1000,
                      'width': w*o.x_scale / 1000,
                      'height': h*o.y_scale / 1000}
            o_list.append(o_dict)

    # bounds
    bounds = {'x': [0, x_bound], 'y': [0, y_bound], 'rotation': [0, 31.4159265359]}
    global_dict['bounds'] = bounds

    # obstacles
    global_dict['obstacles'] = o_list

    # number of obstacles
    global_dict['n_obstacles'] = len(obstacles)-1

    # ball
    ball = {'radius': 0.01,
            'location': [x_bound/3, y_bound], 'linear_velocity': [0.1, -0.05]}
    global_dict['ball'] = ball

    return global_dict


def fitRectangles(pic, visualize=False):

    # calibrate camera first
    cam = cc.Cali_Cam()
    target_pts, image_pts = cam.extract_points(pic)
    cb_origin, cb_diagonal, x_scale, y_scale = cam.CB_bounds(
        image_pts)  # will be used later to mask out the checkerboard

    upper_left = np.array([cb_origin[0], cb_diagonal[1]]).astype(int)
    bottom_right = np.array([cb_diagonal[0], cb_origin[1]]).astype(int)

    H, intrinsic = cam.camera_params(target_pts, image_pts, pic)
    dist_z = H[2, 3]

    # extraction
    gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)

    # bw image, white is region of interest
    t = np.ceil(0.48*256)  # 0.43 might be better for other pics
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
                Obstacle(c, area, moments, x_scale, y_scale, upper_left, dist_z))

    if visualize:
        for obs in obstacles:
            obs.visualize(pic)
            # print ('type: ', obs.type, ' solidity: ', obs.solidity, ' angle: ', obs.angle)
        cv.imshow('thresh', thresh*255)
        cv.imshow("Contours", pic)
        cv.waitKey(0)

    # send into compose env for calculating bounds
    row, col = thresh.shape

    return composeEnv(obstacles, row, col)


if __name__ == "__main__":
    img_names = os.listdir(TEST_IMGS_PATH)
    # imgs = bb1 = [cv.imread(TEST_IMGS_PATH+name) for name in img_names]
    imgs = cv.imread('all2.png')

    environment = fitRectangles(imgs, visualize=True)
    pprint.pprint(environment)
    # r = requests.post('http://localhost:5000/getpose', json=environment)
    # pprint.pprint(r.json())
    pass
