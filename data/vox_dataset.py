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

class VoxDataset(BaseDataset):
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
       
        if opt.isTrain:
            _file = open(os.path.join(self.root, 'pickle','train_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
        else :
            _file = open(os.path.join(self.root, 'pickle','test_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()

        self.data = self.get_files()

        print (len(self.data))
        
        img_params = self.get_img_params(self.output_shape)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BICUBIC)),
            transforms.Lambda(lambda img: self.__color_aug(img, img_params['color_aug'])),
            transforms.Lambda(lambda img: self.__flip(img, img_params['flip'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.transform_L = transforms.Compose([
            transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BILINEAR)),
            transforms.ToTensor()
        ])

    def get_files(self):
        files = []
        for dirs in self.data:
            video = os.path.join(self.root, "unzip/test_video", dirs[0], dirs[1], dirs[2]+"_aligned.mp4")
            lmark = os.path.join(self.root, "unzip/test_video", dirs[0], dirs[1], dirs[2]+"_aligned.npy")
        files.append([lmark, video])
        return files

    def __len__(self):
        return len(self.data) 

    def __color_aug(self, img, params):
        h, s, v = img.convert('HSV').split()    
        h = h.point(lambda i: (i + params[0]) % 256)
        s = s.point(lambda i: min(255, max(0, i * params[1] + params[2])))
        v = v.point(lambda i: min(255, max(0, i * params[3] + params[4])))
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
        return img

    def __flip(self, img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
        
    def __scale_image(self, img, size, method=Image.BICUBIC):
        w, h = size    
        return img.resize((w, h), method)

    def name(self):
        return 'FaceForensicsLmark2rgbDataset'

    def __getitem__(self, index):
        video_path = self.data[index][1] #os.path.join(self.root, 'pretrain', v_id[0] , v_id[1][:5] + '_crop.mp4'  )
        lmark_path  = self.data[index][0]  #= os.path.join(self.root, 'pretrain', v_id[0] , v_id[1]  )

        # read in data
        lmarks = np.load(lmark_path)#[:,:,:-1]
        real_video = self.read_videos(video_path)
        v_length = len(real_video)

        # sample index of frames for embedding network
        input_indexs, target_id = self.get_image_index(self.n_frames_total, v_length)

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
    def get_image_index(self, n_frames_total, cur_seq_len, max_t_step=4):
        if self.opt.isTrain:                
            n_frames_total = min(cur_seq_len, n_frames_total)             # total number of frames to load
            max_t_step = min(max_t_step, (cur_seq_len-1) // max(1, (n_frames_total-1)))        
            t_step = np.random.randint(max_t_step) + 1                    # spacing between neighboring sampled frames                
            
            offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible frame index for the first frame

            start_idx = np.random.randint(offset_max)                 # offset for the first frame to load        
            max_range, min_range = 300, 14                            # range for possible reference frames
            
            # indices for reference frames
            ref_range = list(range(max(0, start_idx - max_range), max(1, start_idx - min_range))) \
                    + list(range(min(start_idx + min_range, cur_seq_len - 1), min(start_idx + max_range, cur_seq_len)))
            ref_indices = np.random.choice(ref_range, size=self.num_frames)   

            # indices for target
            target_ids = [start_idx + step * t_step for step in range(self.n_frames_total)]

        else:
            n_frames_total = 1
            t_step = 1        
            ref_indices = self.opt.ref_img_id.split(',')
            ref_indices = [int(i) for i in ref_indices]
            target_ids = np.random.randint(cur_seq_len)
            while target_ids in ref_indices:
                target_ids = np.random.randint(cur_seq_len)

        if type(target_ids) == list:
            target_ids = target_ids[0]

        return ref_indices, target_ids

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
        input_tensor = transform_L(Image.fromarray(im_edges))
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

    def get_img_params(self, size):
        w, h = size
        
        # for color augmentation
        h_b = random.uniform(-30, 30)
        s_a = random.uniform(0.8, 1.2)
        s_b = random.uniform(-10, 10)
        v_a = random.uniform(0.8, 1.2)
        v_b = random.uniform(-10, 10)    
        
        flip = random.random() > 0.5
        return {'new_size': (w, h), 'flip': flip, 
                'color_aug': (h_b, s_a, s_b, v_a, v_b)}