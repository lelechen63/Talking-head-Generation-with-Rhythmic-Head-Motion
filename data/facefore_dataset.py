import os
from datetime import datetime
import pickle as pkl
import random
import scipy.ndimage.morphology

import PIL
import cv2
import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import mmcv
from io import BytesIO
from PIL import Image
import sys
# sys.path.insert(1, '../utils')
# from .. import utils
from torch.utils.data import DataLoader

from data.base_dataset import BaseDataset, get_transform
from data.keypoint2img import interpPoints, drawEdge

import pdb

class FaceForeDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        parser.add_argument('--aspect_ratio', type=float, default=1)        
        parser.add_argument('--no_upper_face', action='store_true', help='do not add upper face')
        parser.add_argument('--use_ft', action='store_true')

        return parser

    """ Dataset object used to access the pre-processed VoxCelebDataset """
    def initialize(self,opt):
        """
        Instantiates the Dataset.
        :param root: Path to the folder where the pre-processed dataset is stored.
        :param extension: File extension of the pre-processed video files.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
        """
        self.output_shape = tuple([opt.loadSize, opt.loadSize])
        self.num_frames = opt.n_shot
        self.n_frames_total = opt.n_frames_G - 1
        self.opt = opt
        self.root  = opt.dataroot
        self.fix_crop_pos = True

        # mapping from keypoints to face part 
        self.add_upper_face = not opt.no_upper_face
        self.part_list = [[list(range(0, 17)) + ((list(range(68, 83)) + [0]) if self.add_upper_face else [])], # face
                     [range(17, 22)],                                  # right eyebrow
                     [range(22, 27)],                                  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],              # nose
                     [[36,37,38,39], [39,40,41,36]],                   # right eye
                     [[42,43,44,45], [45,46,47,42]],                   # left eye
                     [range(48, 55), [54,55,56,57,58,59,48], range(60, 65), [64,65,66,67,60]], # mouth and tongue
                    ]
       
        # single video try
        _file = open(os.path.join(self.root, 'pickle','train_lmark2img.pkl'), "rb")
        self.data = pkl.load(_file)
        _file.close()

        print (len(self.data))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.transform_L = transforms.Compose([
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.data) 

    
    def name(self):
        return 'FaceForensicsLmark2rgbDataset'

    def __getitem__(self, index):
        video_path = self.data[index][1] #os.path.join(self.root, 'pretrain', v_id[0] , v_id[1][:5] + '_crop.mp4'  )
        lmark_path  = self.data[index][0]  #= os.path.join(self.root, 'pretrain', v_id[0] , v_id[1]  )

        lmarks = np.load(lmark_path)#[:,:,:-1]
        v_length = lmarks.shape[0]

        # sample index of frames for embedding network
        input_indexs, target_id = self.get_image_index(v_length)
        
        # read in all frames in video
        real_video = self.read_videos(video_path)

        # define scale
        self.define_scale()

        # get reference
        ref_images, ref_lmarks = self.prepare_datas(real_video, lmarks, input_indexs)

        # get target
        tgt_images, tgt_lmarks = self.prepare_datas(real_video, lmarks, [target_id])

        target_img_path  = os.path.join(video_path[:-4] , '%05d.png'%target_id  )

        ref_images = torch.cat([ref_img.unsqueeze(0) for ref_img in ref_images], axis=0)
        ref_lmarks = torch.cat([ref_lmark.unsqueeze(0) for ref_lmark in ref_lmarks], axis=0)
        tgt_images = torch.cat([tgt_img.unsqueeze(0) for tgt_img in tgt_images], axis=0)
        tgt_lmarks = torch.cat([tgt_lmark.unsqueeze(0) for tgt_lmark in tgt_lmarks], axis=0)

        input_dic = {'v_id' : target_img_path, 'tgt_label': tgt_lmarks, 'ref_image':ref_images , 'ref_label': ref_lmarks, \
        'tgt_image': tgt_images,  'target_id': target_id}

        return input_dic

    # get index for target and reference
    def get_image_index(self, v_length):
        # sample frames for embedding network
        if self.opt.use_ft:
            if self.num_frames  ==1 :
                input_indexs = [0]
                target_id = 0
            elif self.num_frames == 8:
                input_indexs = [0,7,15,23,31,39,47,55]
                target_id =  random.sample(input_indexs, 1)
                input_indexs = set(input_indexs ) - set(target_id)
                input_indexs =list(input_indexs) 

            elif self.num_frames == 32:
                input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
                target_id =  random.sample(input_indexs, 1)
                input_indexs = set(input_indexs ) - set(target_id)
                input_indexs =list(input_indexs)                    
        else:
            input_indexs  = set(random.sample(range(0,64), self.num_frames))
            # we randomly choose a target frame 
            target_id =  random.randint( 64, v_length - 2)
                
        if type(target_id) == list:
            target_id = target_id[0]

        return list(input_indexs), target_id

    # load in all frames from video
    def read_videos(self, video_path):
        cap = cv2.VideoCapture(video_path)
        real_video = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                real_video.append(frame)
            else:
                break

        return real_video

    # plot landmarks
    def get_face_image(self, keypoints, transform_L, size, bw):   
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        for edge_list in self.part_list:
            for edge in edge_list:
                for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i+edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]
                                    
                    curve_x, curve_y = interpPoints(x, y) # interp keypoints to get the curve shape                    
                    drawEdge(im_edges, curve_x, curve_y, bw=1)        
        input_tensor = transform_L(im_edges)
        return input_tensor

    # preprocess for landmarks
    def get_keypoints(self, keypoints, transform_L, size, crop_coords, bw):
        # crop landmarks
        keypoints[:, 0] -= crop_coords[2]
        keypoints[:, 1] -= crop_coords[0]

        # add upper half face by symmetry
        if self.add_upper_face:
            pts = keypoints[:17, :].astype(np.int32)
            baseline_y = (pts[0,1] + pts[-1,1]) / 2
            upper_pts = pts[1:-1,:].copy()
            upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
            keypoints = np.vstack((keypoints, upper_pts[::-1,:])) 

        # get image from landmarks
        lmark_image = self.get_face_image(keypoints, transform_L, size, bw)

        return lmark_image

    # preprocess for image
    def get_image(self, image, transform_I, size, crop_coords):
        # crop
        img = mmcv.bgr2rgb(image)
        img = self.crop(Image.fromarray(img), crop_coords)

        # transform
        # img = cv2.resize(img, size)
        img = img.resize(size)
        img = transform_I(img)

        return img


    # get scale for random crop
    def define_scale(self, scale_max = 0.2):
        self.scale = [np.random.uniform(1 - scale_max, 1 + scale_max), 
                        np.random.uniform(1 - scale_max, 1 + scale_max)]    

    # get image and landmarks
    def prepare_datas(self, images, lmarks, choice_ids):
        # get cropped coordinates
        crop_lmark = lmarks[choice_ids[0]]
        crop_coords = self.get_crop_coords(crop_lmark)
        bw = max(1, (crop_coords[1]-crop_coords[0]) // 256)

        # get images and landmarks
        result_lmarks = []
        result_images = []
        for choice in choice_ids:
            lmark = self.get_keypoints(lmarks[choice], self.transform_L, self.output_shape, crop_coords, bw)
            image = self.get_image(images[choice], self.transform, self.output_shape, crop_coords)

            result_lmarks.append(lmark)
            result_images.append(image)

        return result_images, result_lmarks

    # get crop standard from one landmark
    def get_crop_coords(self, keypoints, crop_size=None):           
        min_y, max_y = int(keypoints[:,1].min()), int(keypoints[:,1].max())
        min_x, max_x = int(keypoints[:,0].min()), int(keypoints[:,0].max())
        x_cen, y_cen = (min_x + max_x) // 2, (min_y + max_y) // 2                
        w = h = (max_x - min_x)
        if crop_size is not None:
            h, w = crop_size[0] / 2, crop_size[1] / 2
        if self.opt.isTrain and self.fix_crop_pos:
            offset_max = 0.2
            offset = [np.random.uniform(-offset_max, offset_max), 
                      np.random.uniform(-offset_max, offset_max)]             
            w *= self.scale[0]
            h *= self.scale[1]
            x_cen += int(offset[0]*w)
            y_cen += int(offset[1]*h)
                        
        min_x = x_cen - w
        min_y = y_cen - h*1.25
        max_x = min_x + w*2        
        max_y = min_y + h*2

        return int(min_y), int(max_y), int(min_x), int(max_x)