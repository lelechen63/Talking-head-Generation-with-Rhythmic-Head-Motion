# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import re
import torch
import numpy as np
from PIL import Image
import os
import cv2
import math
import pdb

def get_roi_backup(lmark, mask_eyes=True, mask_mouth=True): #lmark shape (68,2) or (68,3) , tempolate shape(256, 256, 1)
    lmark = lmark.copy()
    tempolate = np.zeros((256, 256 , 1), np.uint8)
    eyes =[21, 22, 24,  26, 36, 39,42, 45]
    # eyes =[36, 37, 38, 39, 40, 41, 42, 43, 45, 46]
    eyes_x = []
    eyes_y = []
    for i in eyes:
        eyes_x.append(lmark[i,0])
        eyes_y.append(lmark[i,1])
    min_x = lmark[eyes[np.argmin(eyes_x)], 0] 
    max_x = lmark[eyes[np.argmax(eyes_x)], 0] 
    min_y = lmark[eyes[np.argmin(eyes_y)], 1]
    
    max_y = lmark[eyes[np.argmax(eyes_y)], 1]
    min_x = max(0, int(min_x-5) )
    max_x = min(255, int(max_x+5) )
    min_y = max(0, int(min_y-5) )
    max_y = min(255, int(max_y+5) )

    tempolate[ int(min_y): int(max_y), int(min_x):int(max_x)] = 1 if mask_eyes else 0
    mouth = [48, 50, 51, 54, 57]
    mouth_x = []
    mouth_y = []
    for i in mouth:
        mouth_x.append(lmark[i,0])
        mouth_y.append(lmark[i,1])
    min_x2 = lmark[mouth[np.argmin(mouth_x)], 0] 
    max_x2 = lmark[mouth[np.argmax(mouth_x)], 0] 
    min_y2 = lmark[mouth[np.argmin(mouth_y)], 1]
    max_y2 = lmark[mouth[np.argmax(mouth_y)], 1]  

    min_x2 = max(0, int(min_x2-5) )
    max_x2 = min(255, int(max_x2+5) )
    min_y2 = max(0, int(min_y2-5) )
    max_y2 = min(255, int(max_y2+5) )

    
    tempolate[int(min_y2):int(max_y2), int(min_x2):int(max_x2)] = 1 if mask_mouth else 0
    return  tempolate

def get_roi_small_eyes(lmark, mask_eyes=True, mask_mouth=True): #lmark shape (68,2) or (68,3) , tempolate shape(256, 256, 1)
    lmark = lmark.copy()
    tempolate = np.zeros((256, 256 , 1), np.uint8)
    # eyes =[21, 22, 24,  26, 36, 39,42, 45]
    l_eyes =[36, 37, 38, 39, 40, 41]
    r_eyes = [42, 43, 44, 45, 46, 47]
    l_eyes_x = []
    l_eyes_y = []
    for i in l_eyes:
        l_eyes_x.append(lmark[i,0])
        l_eyes_y.append(lmark[i,1])
    min_x = lmark[l_eyes[np.argmin(l_eyes_x)], 0] 
    max_x = lmark[l_eyes[np.argmax(l_eyes_x)], 0] 
    min_y = lmark[l_eyes[np.argmin(l_eyes_y)], 1]
    
    max_y = lmark[l_eyes[np.argmax(l_eyes_y)], 1]
    min_x = max(0, int(min_x-2) )
    max_x = min(255, int(max_x+2) )
    min_y = max(0, int(min_y-2) )
    max_y = min(255, int(max_y+2) )

    tempolate[ int(min_y): int(max_y), int(min_x):int(max_x)] = 0

    r_eyes_x = []
    r_eyes_y = []
    for i in r_eyes:
        r_eyes_x.append(lmark[i,0])
        r_eyes_y.append(lmark[i,1])
    min_x = lmark[r_eyes[np.argmin(r_eyes_x)], 0] 
    max_x = lmark[r_eyes[np.argmax(r_eyes_x)], 0] 
    min_y = lmark[r_eyes[np.argmin(r_eyes_y)], 1]
    
    max_y = lmark[r_eyes[np.argmax(r_eyes_y)], 1]
    min_x = max(0, int(min_x-2) )
    max_x = min(255, int(max_x+2) )
    min_y = max(0, int(min_y-2) )
    max_y = min(255, int(max_y+2) )

    tempolate[ int(min_y): int(max_y), int(min_x):int(max_x)] = 1
    
    return  tempolate

def get_roi(lmark, mask_eyes=True, mask_mouth=True): #lmark shape (68,2) or (68,3) , tempolate shape(256, 256, 1)
    lmark = lmark.copy()
    tempolate = np.zeros((256, 256 , 1), np.uint8)
    # eyes =[21, 22, 24,  26, 36, 39,42, 45]
    l_eyes =[36, 37, 38, 39, 40, 41]
    r_eyes = [42, 43, 44, 45, 46, 47]
    l_eyes_x = []
    l_eyes_y = []
    for i in l_eyes:
        l_eyes_x.append(lmark[i,0])
        l_eyes_y.append(lmark[i,1])
    min_x = lmark[l_eyes[np.argmin(l_eyes_x)], 0] 
    max_x = lmark[l_eyes[np.argmax(l_eyes_x)], 0] 
    min_y = lmark[l_eyes[np.argmin(l_eyes_y)], 1]
    
    max_y = lmark[l_eyes[np.argmax(l_eyes_y)], 1]
    min_x = max(0, int(min_x-2) )
    max_x = min(255, int(max_x+2) )
    min_y = max(0, int(min_y-2) )
    max_y = min(255, int(max_y+2) )

    tempolate[ int(min_y): int(max_y), int(min_x):int(max_x)] = 1 if mask_eyes else 0

    r_eyes_x = []
    r_eyes_y = []
    for i in r_eyes:
        r_eyes_x.append(lmark[i,0])
        r_eyes_y.append(lmark[i,1])
    min_x = lmark[r_eyes[np.argmin(r_eyes_x)], 0] 
    max_x = lmark[r_eyes[np.argmax(r_eyes_x)], 0] 
    min_y = lmark[r_eyes[np.argmin(r_eyes_y)], 1]
    
    max_y = lmark[r_eyes[np.argmax(r_eyes_y)], 1]
    min_x = max(0, int(min_x-2) )
    max_x = min(255, int(max_x+2) )
    min_y = max(0, int(min_y-2) )
    max_y = min(255, int(max_y+2) )

    tempolate[ int(min_y): int(max_y), int(min_x):int(max_x)] = 1 if mask_eyes else 0
    
    mouth = [48, 50, 51, 54, 57]
    mouth_x = []
    mouth_y = []
    for i in mouth:
        mouth_x.append(lmark[i,0])
        mouth_y.append(lmark[i,1])
    min_x2 = lmark[mouth[np.argmin(mouth_x)], 0] 
    max_x2 = lmark[mouth[np.argmax(mouth_x)], 0] 
    min_y2 = lmark[mouth[np.argmin(mouth_y)], 1]
    max_y2 = lmark[mouth[np.argmax(mouth_y)], 1]  

    min_x2 = max(0, int(min_x2-5) )
    max_x2 = min(255, int(max_x2+5) )
    min_y2 = max(0, int(min_y2-5) )
    max_y2 = min(255, int(max_y2+5) )

    
    tempolate[int(min_y2):int(max_y2), int(min_x2):int(max_x2)] = 1 if mask_mouth else 0
    return  tempolate

def get_abso_mouth(lmark):
    left_x, left_y = lmark[48][:-1]
    right_x, right_y = lmark[54][:-1]
    mid_x = (left_x+right_x)/2.0
    mid_y = (left_y+right_y)/2.0

    left = max(0, int(mid_x-40))
    right = min(255, int(mid_x+40))
    up = max(0, int(mid_y-40))
    bottom = min(255, int(mid_y+40))

    template = np.zeros((256, 256 , 1), np.uint8)
    template[up:bottom, left:right] = 1

    return template


def visualize_label(opt, label_tensor, model=None): 
    if label_tensor.dim() == 5:
        label_tensor = label_tensor[-1]
    if label_tensor.dim() == 4:        
        label_tensor = label_tensor[-1]
    if opt.label_nc:
        visual_label = tensor2label(label_tensor[:opt.label_nc], opt.label_nc)
    else:
        visual_label = tensor2im(label_tensor[:3] if label_tensor.shape[0] >= 3 else label_tensor[:1])

    if len(visual_label.shape) == 2: visual_label = np.repeat(visual_label[:,:,np.newaxis], 3, axis=2)        
    return visual_label

# Converts a Tensor into a Numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if image_tensor is None: return None
    if isinstance(image_tensor, list):
        image_tensor = [t for t in image_tensor if t is not None]
        if not image_tensor: return None
        images_np = [tensor2im(t, imtype, normalize) for t in image_tensor]
        return tile_images(images_np) if tile else images_np
    
    if image_tensor.dim() == 5:
        image_tensor = image_tensor[-1]
    if image_tensor.dim() == 4:
        if tile:            
            images_np = [tensor2im(image_tensor[b]) for b in range(image_tensor.size(0))]
            return tile_images(images_np)
        image_tensor = image_tensor[-1]
    elif image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)

    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        #image_numpy = image_numpy[:,:,0]
        image_numpy = np.repeat(image_numpy, 3, axis=2)
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):    
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result

def tensor2flow(tensor, imtype=np.uint8, tile=False):
    if tensor is None: return None
    if isinstance(tensor, list):
        tensor = [t for t in tensor if t is not None]
        if not tensor: return None
        images_np = [tensor2flow(t, imtype) for t in tensor]        
        return tile_images(images_np) if tile else images_np        
    if tensor.dim() == 5:
        tensor = tensor[-1]
    if tensor.dim() == 4:
        if tile:
            images_np = [tensor2flow(tensor[b]) for b in range(tensor.size(0))]
            return tile_images(images_np)        
        tensor = tensor[-1]
    tensor = tensor.detach().cpu().float().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    hsv = np.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(tensor[..., 0], tensor[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def tile_images(imgs, picturesPerRow=2):
    """ Convert to a true list of 16x16 images"""
    if type(imgs) == list:
        if len(imgs) == 1: return imgs[0]
        imgs = [img[np.newaxis,:] for img in imgs]
        imgs = np.concatenate(imgs, axis=0)

    # Calculate how many columns
    #picturesPerColumn = imgs.shape[0]/picturesPerRow + 1*((imgs.shape[0]%picturesPerRow)!=0)
    #picturesPerColumn = int(picturesPerColumn)
    
    # Padding
    #rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow        
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)    

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):        
        tiled.append(np.concatenate([imgs[j] for j in range(i, i+picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled
    
def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

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
    if N == 35: # cityscape train
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                    dtype=np.uint8)
    elif N == 20: # GTA/cityscape eval
        cmap = np.array([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), (250,170, 30), 
                         (220,220,  0), (107,142, 35), (152,251,152), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70), 
                         (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), ( 70,130,180), (  0,  0,  0)], 
                         dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1 # let's give 0 a color
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

#  calculate mouth open
def openrate(lmark1):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    for k in range(3):
        open_rate1.append(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2])
        
    open_rate1 = np.asarray(open_rate1)
    return open_rate1.mean() 

def eye_blinking(lmark, rate = 10): #lmark shape (k, 68,2) or (k,68,3) , tempolate shape(256, 256, 1)
    length = lmark.shape[0]
    bink_time = math.floor(length / float(rate) )
    eys =[[37,41],[38,40] ,[43,47],[44,46]]  # [upper, lower] , [left1,left2, right1, right1]
    for i in range(bink_time):
        print ('+++++')
        for e in eys:
            dis =  (np.abs(lmark[0, e[0],:2] -  lmark[0, e[1],:2] ) / 2)
            print ('--------')
            # -2
            lmark[rate * (i + 1)-2, e[0],:2] += 0.45 * (dis)
            lmark[rate * (i + 1)-2, e[1],:2] -= 0.45 * (dis)
            # +2
            lmark[rate * (i + 1)+2, e[0], :2] += 0.45 * (dis)
            lmark[rate * (i + 1)+2, e[1], :2] -= 0.45 * (dis)
            # -1
            lmark[rate * (i + 1)-1, e[0], :2] += 0.85 * (dis)
            lmark[rate * (i + 1)-1, e[1], :2] -= 0.85 * (dis)
            # +1
            lmark[rate * (i + 1)+1, e[0], :2] += 0.8 * (dis)
            lmark[rate * (i + 1)+1, e[1], :2] -= 0.8 * (dis)
            # 0
            lmark[rate * (i + 1), e[0], :2] += 0.95 * (dis)
            lmark[rate * (i + 1), e[1], :2] -= 0.95 * (dis)
    return lmark