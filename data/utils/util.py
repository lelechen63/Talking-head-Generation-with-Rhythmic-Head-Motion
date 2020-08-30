import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import argparse, os, fnmatch, shutil
from collections import OrderedDict
from scipy.spatial import procrustes
from skimage import io
import matplotlib.animation as manimation
import matplotlib.lines as mlines
import numpy as np
import cv2
import math
import copy
import librosa
import subprocess
# from keras import backend as K
from tqdm import tqdm
from skimage import transform as tf
from PIL import Image

from scipy.spatial.transform import Rotation as R

font = {'size'   : 18}
mpl.rc('font', **font)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)



def read_videos( video_path):
    cap = cv2.VideoCapture(video_path)
    real_video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            real_video.append(frame)
        else:
            break

    return real_video

def rt_to_degree( RT ):
    #   RT (6,)
    RT = np.mat(RT)
    # recover the transformation
    rec = RT[0, :3]
    r = R.from_rotvec(rec)
    # print (r)
    ret_R =  r.as_euler('zyx', degrees=True)
    # print (ret_R)
    return ret_R
 

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def plot_landmarks( landmarks):
    # landmarks = np.int32(landmarks)
    blank_image = np.zeros((256,256,3), np.uint8) 

    # cv2.polylines(blank_image, np.int32([points]), True, (0,255,255), 1)

    cv2.polylines(blank_image, np.int32([landmarks[0:17]]) , True, (0,255,255), 2)
 
    cv2.polylines(blank_image,  np.int32([landmarks[17:22]]), True, (255,0,255), 2)

    cv2.polylines(blank_image, np.int32([landmarks[22:27]]) , True, (255,0,255), 2)

    cv2.polylines(blank_image, np.int32([landmarks[27:31]]) , True, (255,255, 0), 2)

    cv2.polylines(blank_image, np.int32([landmarks[31:36]]) , True, (255,255, 0), 2)

    cv2.polylines(blank_image, np.int32([landmarks[36:42]]) , True, (255,0, 0), 2)
    cv2.polylines(blank_image, np.int32([landmarks[42:48]]) , True, (255,0, 0), 2)

    cv2.polylines(blank_image, np.int32([landmarks[48:60]]) , True, (0, 0, 255), 2)

    return blank_image
    
def smooth(kps, ALPHA1=0.2, ALPHA2=0.7):
    
    n = kps.shape[0]

    kps_new = np.zeros_like(kps)

    for i in range(n):
        if i==0:
            kps_new[i,:,:] = kps[i,:,:]
        else:
            kps_new[i,:48,:] = ALPHA1 * kps[i,:48,:] + (1-ALPHA1) * kps_new[i-1,:48,:]
            kps_new[i,48:,:] = ALPHA2 * kps[i,48:,:] + (1-ALPHA2) * kps_new[i-1,48:,:]

    # np.save(out_file, kps_new)
    return kps_new

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

# Lookup tables for drawing lines between points
Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
         [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
         [66, 67], [67, 60]]

Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33], \
        [33, 34], [34, 35], [27, 31], [27, 35]]

leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]

leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
         [12, 13], [13, 14], [14, 15], [15, 16]]

faceLmarkLookup = Mouth + Nose + leftBrow + rightBrow + leftEye + rightEye + other



def oned_smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size."
        )

    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y




def crop_mouth(img, lmark):
    (x, y, w, h) = cv2.boundingRect(lmark[48:68,:-1].astype(int))

    center_x = x + int(0.5 * w)

    center_y = y + int(0.5 * h)

    r = 32
    new_x = center_x - r
    new_y = center_y - r
    roi = img[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
    return roi


def mse_metrix(lmark1, lmark2):
    #input shape (68,3)
    
    distance =  np.square(lmark1 - lmark2)
    if distance.shape == (68,3):
        return distance[:,:-1].mean()
    else:
        return distance.mean()


def openrate(lmark1):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    for k in range(3):
        open_rate1.append(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2])
        
    open_rate1 = np.asarray(open_rate1)
    return open_rate1.mean() 
        

def openrate_metrix(lmark1, lmark2):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    open_rate2 = []
    for k in range(3):
        open_rate1.append(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2])
        
        
        open_rate2.append(lmark2[open_pair[k][0],:2] - lmark2[open_pair[k][1], :2])
        
    open_rate1 = np.asarray(open_rate1)
    open_rate2 = np.asarray(open_rate2)
    return mse_metrix(open_rate1, open_rate2)  
        
    #input shape (68,3)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def reverse_rt(source,  RT):
    #source (68,3) , RT (6,)
    source =  np.mat(source)
    RT = np.mat(RT)
    # recover the transformation
    rec = RT[0,:3]
    r = R.from_rotvec(rec)
    ret_R = r.as_dcm()
    ret_R2 = ret_R[0].T
    ret_t = RT[0,3:]
    ret_t = ret_t.reshape(3,1)
    ret_t2 = - ret_R2 * ret_t
    ret_t2 = ret_t2.reshape(3,1)
    A3 = ret_R2 *   source.T +  np.tile(ret_t2, (1,68))
    A3 = A3.T
    
    return A3

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def lmark2img( lmark,img= None, c = 'w'):        
    preds = lmark
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 1, 1)
    #if img != None:
#         img  = io.imread(img_path)
    ax.imshow(img)
    ax.plot(preds[0:17,0],preds[0:17,1]  ,marker='o',markersize=1,linestyle='-',color=c,lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color=c,lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color=c,lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color= c,lw=1) 
    ax.axis('off')
    ax.set_xlim(ax.get_xlim()[::-1])
    
    return plt


def compare_vis(img,lmark1,lmark2):
#     img  = io.imread(img_path)
    preds = lmark1
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
    ax.axis('off')
    
    preds = lmark2
    
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
    ax.axis('off')
    return plt



def plot_flmarks(pts, lab, xLim, yLim, xLab, yLab, figsize=(10, 10), sentence = None):
    if len(pts.shape) != 2:
        pts = pts.reshape( 68, 2)

    if pts.shape[0] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print (lookup)
    else:
        lookup = faceLmarkLookup

    plt.figure(figsize=figsize)
    plt.plot(pts[:,0], pts[:,1], 'ko', ms=4)
    for refpts in lookup:
        plt.plot([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]], 'k', ms=4)


    # pts = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front.npy')
    # if len(pts.shape) != 2:
    #     pts = pts.reshape( 68, 2)
    # lookup = faceLmarkLookup
    # plt.plot(pts[:,0], pts[:,1], 'go', ms=4)
    # for refpts in lookup:
    #     plt.plot([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]], 'g', ms=4)


    plt.xlabel(xLab, fontsize = font['size'] + 4, fontweight='bold')
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top') 
    plt.ylabel(yLab, fontsize = font['size'] + 4, fontweight='bold')
    plt.xlim(xLim)
    plt.ylim(yLim)
    
    plt.gca().invert_yaxis()
    if sentence is not None:
        plt.xlabel(sentence)
    plt.savefig(lab, dpi = 100, bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_flmarks3D(pts, lab, xLim, yLim, zLim,rotate=False,  figsize=(10, 10), sentence =None):
    pts = np.reshape(pts, (68, 3))

    if pts.shape[0] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print (lookup)
    else:
        lookup = faceLmarkLookup

    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    l, = ax.plot3D([], [], [], 'ko', ms=2)

    lines = [ax.plot([], [], [], 'k', lw=1)[0] for _ in range(3*len(lookup))]
    ax.set_xlim3d(xLim)     
    ax.set_ylim3d(yLim)     
    ax.set_zlim3d(zLim)
    ax.set_xlabel('x', fontsize=28)
    ax.set_ylabel('y', fontsize=28)
    ax.set_zlabel('z', fontsize=28)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    if rotate:
            angles = np.linspace(60, 120,1)
    else:
        angles = np.linspace(60, 60, 1)

    ax.view_init(elev=60, azim=angles[0])
    l.set_data(pts[:,0], pts[:,1])
    l.set_3d_properties(pts[:,2])
    cnt = 0
    for refpts in lookup:
        lines[cnt].set_data([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]])
        lines[cnt].set_3d_properties([pts[ refpts[1], 2], pts[refpts[0], 2]])
        cnt+=1
    if sentence is not None:
        plt.xlabel(sentence)
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()

def melSpectra(y, sr, wsize, hsize):
    cnst = 1+(int(sr*wsize)/2)
    y_stft_abs = np.abs(librosa.stft(y,
                                  win_length = int(sr*wsize),
                                  hop_length = int(sr*hsize),
                                  n_fft=int(sr*wsize)))/cnst

    melspec = np.log(1e-16 + librosa.feature.melspectrogram(sr=sr, 
                                             S=y_stft_abs**2,
                                             n_mels=64))
    return melspec


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


## Covert a landmark Tensor into image numpy
def lmark2im(lmark_tensor, imtype=np.uint8, normalize=True):
    if isinstance(lmark_tensor, list):
        image_numpy = []
        for i in range(len(lmark_tensor)):
            image_numpy.append(lmark2im(lmark_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = lmark_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    
    print (image_numpy)
    
    
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 1)


###############################################################################


def crop_image(image_path, detector, shape, predictor):
    

  
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
      
        (x, y, w, h) = rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)

        r = int(0.64 * h)
        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        
        roi = cv2.resize(roi, (163,163), interpolation = cv2.INTER_AREA)
        scale =  163. / (2 * r)
       
        shape = ((shape - np.array([new_x,new_y])) * scale)
    
        return roi, shape 

      
def image_to_video(sample_dir = None, video_name = None):
    
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.png -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 
    #ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
    print (command)
    os.system(command)

def add_audio(video_name=None, audio_dir = None):

    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
    #ffmpeg -i /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/new/audio/obama.wav -codec copy -c:v libx264 -c:a aac -b:a 192k  -shortest -y /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mov
    # ffmpeg -i gan_r_high_fake.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/audio/obama.wav -vcodec copy  -acodec copy -y   gan_r_high_fake.mov

    print (command)
    os.system(command)

def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx




def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale
    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    path_to_file = os.path.join('../results', file_name + '.jpg')
    # Convert RBG to GBR
    gradient = gradient[..., ::-1]
    cv2.imwrite(path_to_file, gradient)


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    path_to_file = os.path.join('./results', file_name+'_Cam_Grayscale.jpg')
    print (path_to_file)
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('./results', file_name+'_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (128, 128))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('./results', file_name+'_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))



def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency

def write_video( frames, sound, fs, path, fname, xLim, yLim):
        try:
            os.remove(os.path.join(path, fname+'.mp4'))
            os.remove(os.path.join(path, fname+'.wav'))
            os.remove(os.path.join(path, fname+'_ws.mp4'))
        except:
            print ('Exp')

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=25, metadata=metadata)

        fig = plt.figure(figsize=(10, 10))
        # l, = plt.plot([], [], 'ko', ms=4)

        # plt.xlim(xLim)
        # plt.ylim(yLim)

        librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

        with writer.saving(fig, os.path.join(path, fname+'.mp4'), 100):
            # plt.gca().invert_yaxis()
            for i in tqdm(range(frames.shape[0])):
                self.plot_face(frames[i, :, :])
                writer.grab_frame()
                plt.clf()
                # plt.close()

        cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_.mp4'
        subprocess.call(cmd, shell=True) 
        print('Muxing Done')

        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))



class facePainter():
    inds_mouth = [60, 61, 62, 63, 64, 65, 66, 67, 60]
    inds_top_teeth = [48, 54, 53, 52, 51, 50, 49, 48]
    inds_bottom_teeth = [4, 12, 10, 6, 4]
    inds_skin = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                    57, 58, 59, 48, 49, 50, 51, 52, 52, 53, 54, 55, 56, 57,
                    8, 9, 10, 11, 12, 13, 14, 15, 16,
                    45, 46, 47, 42, 43, 44, 45,
                    16, 71, 70, 69, 68, 0,
                    36, 37, 38, 39, 40, 41, 36, 0]
    inds_lips = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48,
                    60, 67, 66, 65, 64, 63, 62, 61, 60, 48]
    inds_nose = [[27, 28, 29, 30, 31, 27],
                    [30, 31, 32, 33, 34, 35, 30],
                    [27, 28, 29, 30, 35, 27]]
    inds_brows = [[17, 18, 19, 20, 21],
                    [22, 23, 24, 25, 26]]

    def __init__(self, lmarks, speech):
        lmarks = np.concatenate((lmarks,
                                lmarks[:, [17, 19, 24, 26], :]), 1)[..., :2]
        lmarks[:, -4:, 1] += -0.03
        # lm = lmarks.mean(0)
        # lm = lmarks[600]

        self.lmarks = lmarks
        self.speech = speech

    def plot_face(self, lm):
        plt.axes().set_aspect('equal', 'datalim')

        # make some eyes
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.transpose([np.cos(theta), np.sin(theta)])
        for self.inds_eye in [[37, 38, 40, 41], [43, 44, 46, 47]]:
            plt.fill(.013 * circle[:, 0] + lm[self.inds_eye, 0].mean(),
                    .013 * circle[:, 1] - lm[self.inds_eye, 1].mean(),
                    color=[0, 0.5, 0], lw=0)
            plt.fill(.005 * circle[:, 0] + lm[self.inds_eye, 0].mean(),
                    .005 * circle[:, 1] - lm[self.inds_eye, 1].mean(),
                    color=[0, 0, 0], lw=0)
        plt.plot(.01 * circle[:, 0], .01 * circle[:, 1], color=[0, 0.5, 0], lw=0)
        # make the teeth
        # nose bottom to top teeth: 0.037
        # chin bottom to bottom teeth: .088
        plt.fill(lm[self.inds_mouth, 0], -lm[self.inds_mouth, 1], color=[0, 0, 0], lw=0)
        # plt.fill(lm[inds_top_teeth, 0], -lm[inds_top_teeth, 1], color=[1, 1, 0.95], lw=0)
        # plt.fill(lm[inds_bottom_teeth, 0], -lm[inds_bottom_teeth, 1], color=[1, 1, 0.95], lw=0)

        # make the rest
        skin_color = np.array([0.7, 0.5, 0.3])
        plt.fill(lm[self.inds_skin, 0], -lm[self.inds_skin, 1], color=skin_color, lw=0)
        for ii, color_shift in zip(self.inds_nose, [-0.05, -0.1, 0.06]):
            plt.fill(lm[ii, 0], -lm[ii, 1], color=skin_color + color_shift, lw=0)
        plt.fill(lm[self.inds_lips, 0], -lm[self.inds_lips, 1], color=[0.7, 0.3, 0.2], lw=0)

        for ib in self.inds_brows:
            plt.plot(lm[ib, 0], -lm[ib, 1], color=[0.3, 0.2, 0.05], lw=4)

        plt.xlim(-0.15, 0.15)
        plt.ylim(-0.2, 0.18)

    def write_video(self, frames, sound, fs, path, fname, xLim, yLim):
        try:
            os.remove(os.path.join(path, fname+'.mp4'))
            os.remove(os.path.join(path, fname+'.wav'))
            os.remove(os.path.join(path, fname+'_ws.mp4'))
        except:
            print ('Exp')

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=25, metadata=metadata)

        fig = plt.figure(figsize=(10, 10))
        # l, = plt.plot([], [], 'ko', ms=4)

        # plt.xlim(xLim)
        # plt.ylim(yLim)

        librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

        with writer.saving(fig, os.path.join(path, fname+'.mp4'), 100):
            # plt.gca().invert_yaxis()
            for i in tqdm(range(frames.shape[0])):
                self.plot_face(frames[i, :, :])
                writer.grab_frame()
                plt.clf()
                # plt.close()

        cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_.mp4'
        subprocess.call(cmd, shell=True) 
        print('Muxing Done')

        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))

    def paintFace(self, path, fname):
        self.write_video(self.lmarks, self.speech, 8000, path, fname, [-0.15, 0.15], [-0.2, 0.18])

def main():
    return

if __name__ == "__main__":
    main()