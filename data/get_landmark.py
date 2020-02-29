import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
import warnings
warnings.simplefilter('ignore')

import pdb

def func(x, a, b, c):    
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def get_part():
    part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]], # face
                     [range(17, 22)],                                  # right eyebrow
                     [range(22, 27)],                                  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],              # nose
                     [[36,37,38,39], [39,40,41,36]],                   # right eye
                     [[42,43,44,45], [45,46,47,42]],                   # left eye
                     [range(48, 55), [54,55,56,57,58,59,48], range(60, 65), [64,65,66,67,60]], # mouth and tongue
                    ]
    
    return part_list

# preprocess for landmarks
def get_keypoints(keypoints, size, bw=1):
    # add upper half face by symmetry
    pts = keypoints[:17, :].astype(np.int32)
    baseline_y = (pts[0,1] + pts[-1,1]) / 2
    upper_pts = pts[1:-1,:].copy()
    upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
    keypoints = np.vstack((keypoints, upper_pts[::-1,:])) 

    # get image from landmarks
    lmark_image = get_face_image(keypoints, size, bw)

    return lmark_image

# plot landmarks
def get_face_image(keypoints, size, bw):   
    w, h = size

    edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
    for edge_list in get_part():
        for edge in edge_list:
            for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                sub_edge = edge[i:i+edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]
                                
                curve_x, curve_y = interpPoints(x, y) # interp keypoints to get the curve shape                    
                drawEdge(im_edges, curve_x, curve_y, bw=bw)        
    lmark_image = Image.fromarray(im_edges)
    return lmark_image

# set color for landmark
def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():            
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]            
        else:            
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]

# draw edge between landmark points
def drawEdge(im, x, y, bw=1, color=(255,255,255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw*2, bw*2):
                for j in range(-bw*2, bw*2):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(0, np.minimum(h-1, np.array([y[0], y[-1]])+i))
                        xx = np.maximum(0, np.minimum(w-1, np.array([x[0], x[-1]])+j))
                        setColor(im, yy, xx, color)

def interpPoints(x, y):    
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y)
            else:
                popt, _ = curve_fit(func, x, y)
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)

if __name__ == '__main__':
    keypoints_p = '/home/cxu-serve/p1/common/grid/align/s1/lbaq6p_original.npy'
    keypoints = np.load(keypoints_p)
    pdb.set_trace()
    lmark_img = get_keypoints(keypoints[0], (256, 256))
    pdb.set_trace()
    print('a')