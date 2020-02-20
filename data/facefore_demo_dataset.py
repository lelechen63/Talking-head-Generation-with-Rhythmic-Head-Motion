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
from scipy.spatial.transform import Rotation as R

from data.base_dataset import BaseDataset, get_transform
from data.keypoint2img import interpPoints, drawEdge

import pdb

class FaceForeDemoDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        parser.add_argument('--aspect_ratio', type=float, default=1)        
        parser.add_argument('--no_upper_face', action='store_true', help='do not add upper face')
        parser.add_argument('--use_ft', action='store_true')
        parser.add_argument('--dataset_name', type=str, help='face or vox or grid')

        # for reference
        parser.add_argument('--ref_img_id', type=str)
        parser.add_argument('--tgt_video_path', type=str)
        parser.add_argument('--tgt_lmarks_path', type=str)
        parser.add_argument('--tgt_rt_path', type=str, default=None)
        parser.add_argument('--tgt_ani_path', type=str, default=None)
        parser.add_argument('--ref_video_path', type=str)
        parser.add_argument('--ref_lmarks_path', type=str)
        parser.add_argument('--ref_rt_path', type=str, default=None)
        parser.add_argument('--ref_front_path', type=str, default=None)
        parser.add_argument('--ref_ani_id', type=int)

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
        assert not opt.isTrain
        self.output_shape = tuple([opt.loadSize, opt.loadSize])
        self.num_frames = opt.n_shot
        self.n_frames_total = opt.n_frames_G
        self.opt = opt
        self.root  = opt.dataroot
        self.fix_crop_pos = True
        self.ref_search = opt.ref_rt_path is not None and opt.tgt_rt_path is not None

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
       
        # load path
        self.tgt_video_path = opt.tgt_video_path
        self.tgt_lmarks_path = opt.tgt_lmarks_path
        self.tgt_ani_path = opt.tgt_ani_path
        self.tgt_rt_path = opt.tgt_rt_path
        self.ref_video_path = opt.ref_video_path
        self.ref_lmarks_path = opt.ref_lmarks_path
        self.ref_rt_path = opt.ref_rt_path
        self.ref_front_path = opt.ref_front_path

        # read in data
        self.tgt_lmarks = np.load(self.tgt_lmarks_path) #[:,:,:-1]
        self.tgt_video = self.read_videos(self.tgt_video_path)
        self.ref_lmarks = np.load(self.ref_lmarks_path)
        self.ref_video = self.read_videos(self.ref_video_path)
        if self.opt.warp_ani:
            self.tgt_ani_video = self.read_videos(self.tgt_ani_path)
            self.ref_front = np.load(self.ref_front_path)
            self.ref_ani_id = self.opt.ref_ani_id
        if self.opt.warp_ani or self.ref_search:
            self.tgt_rt = np.load(self.tgt_rt_path)
            self.ref_rt = np.load(self.ref_rt_path)

        # clean
        correct_nums = self.clean_lmarks(self.tgt_lmarks)
        self.tgt_lmarks = self.tgt_lmarks[correct_nums]
        self.tgt_video = np.asarray(self.tgt_video)[correct_nums]
        if self.opt.warp_ani:
            self.tgt_ani_video = np.asarray(self.tgt_ani_video)[correct_nums]
        if self.opt.warp_ani or self.ref_search:
            self.tgt_rt = self.tgt_rt[correct_nums]

        correct_nums = self.clean_lmarks(self.ref_lmarks)
        self.ref_lmarks = self.ref_lmarks[correct_nums]
        self.ref_video = np.asarray(self.ref_video)[correct_nums]
        if self.opt.warp_ani or self.ref_search:
            self.ref_rt = self.ref_rt[correct_nums]

        # get transform for image and landmark
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self.__scale_image(img, self.output_shape, Image.BICUBIC)),
            transforms.Lambda(lambda img: self.__color_aug(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.transform_L = transforms.Compose([
            transforms.Lambda(lambda img: self.__scale_image(img, self.output_shape, Image.BILINEAR)),
            transforms.ToTensor()
        ])

        # define parameters for inference
        self.ref_video, self.ref_lmarks, self.ref_indices = self.define_inference(self.ref_video, self.ref_lmarks)


    def __len__(self):
        return len(self.tgt_video) 

    def __scale_image(self, img, size, method=Image.BICUBIC):
        w, h = size    
        return img.resize((w, h), method)

    def __color_aug(self, img):
        h, s, v = img.convert('HSV').split()    
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
        return img

    def name(self):
        return 'FaceForensicsDemoDataset'

    def __getitem__(self, index):
        # get target
        target_id = [index]
        tgt_images, tgt_lmarks = self.prepare_datas(self.tgt_video, self.tgt_lmarks, target_id)

        # get animation
        if self.opt.warp_ani:
            # get animation & get cropped ground truth
            ani_lmarks = []
            ani_images = []

            for gg in target_id:
                ani_lmarks.append(self.reverse_rt(self.ref_front[int(self.ref_ani_id)], self.tgt_rt[gg]))
                ani_lmarks[-1] = np.array(ani_lmarks[-1])
                ani_images.append(self.tgt_ani_video[gg])

            ani_images, ani_lmarks = self.prepare_datas(ani_images, ani_lmarks, list(range(len(target_id))))

        # get warping reference
        if self.ref_search:
            ref_rt = self.ref_rt[:, :3]
            tgt_rt = self.tgt_rt[:, :3]
            warping_ref_ids = self.get_warp_ref(tgt_rt, ref_rt, self.ref_indices, target_id)
            warping_refs = [self.ref_video[w_ref_id] for w_ref_id in warping_ref_ids]
            warping_ref_lmarks = [self.ref_lmarks[w_ref_id] for w_ref_id in warping_ref_ids]

        target_img_path = [os.path.join(self.tgt_video_path[:-4] , '%05d.png'%t_id) for t_id in target_id]

        tgt_images = torch.cat([tgt_img.unsqueeze(0) for tgt_img in tgt_images], axis=0)
        tgt_lmarks = torch.cat([tgt_lmark.unsqueeze(0) for tgt_lmark in tgt_lmarks], axis=0)
        if self.opt.warp_ani:
            ani_images = torch.cat([ani_image.unsqueeze(0) for ani_image in ani_images], axis=0)
            ani_lmarks = torch.cat([ani_lmark.unsqueeze(0) for ani_lmark in ani_lmarks], axis=0)
        if self.ref_search:
            warping_refs = torch.cat([warping_ref.unsqueeze(0) for warping_ref in warping_refs], axis=0)
            warping_ref_lmarks = torch.cat([warping_ref_lmark.unsqueeze(0) for warping_ref_lmark in warping_ref_lmarks], axis=0)

        input_dic = {'path': self.tgt_video_path, 'v_id' : target_img_path, 'index':target_id, 'tgt_label': tgt_lmarks, \
            'ref_image':self.ref_video , 'ref_label': self.ref_lmarks, \
            'tgt_image': tgt_images,  'target_id': target_id}
        if self.opt.warp_ani:
            input_dic.update({'ani_image': ani_images, 'ani_lmark': ani_lmarks})
        if self.ref_search:
            input_dic.update({'warping_ref': warping_refs, 'warping_ref_lmark': warping_ref_lmarks})

        return input_dic

    # define parameters for inference
    def define_inference(self, real_video, lmarks):
        # get reference index
        ref_indices = self.opt.ref_img_id.split(',')
        ref_indices = [int(i) for i in ref_indices]

        # get face image
        ref_images, ref_lmarks = self.prepare_datas(real_video, lmarks, ref_indices)

        # concatenate
        ref_images = torch.cat([ref_img.unsqueeze(0) for ref_img in ref_images], axis=0)
        ref_lmarks = torch.cat([ref_lmark.unsqueeze(0) for ref_lmark in ref_lmarks], axis=0)

        return ref_images, ref_lmarks, ref_indices

    # clean landmarks and images
    def clean_lmarks(self, lmarks):
        min_x, max_x = lmarks[:,:,0].min(axis=1).astype(int), lmarks[:,:,0].max(axis=1).astype(int)
        min_y, max_y = lmarks[:,:,1].min(axis=1).astype(int), lmarks[:,:,1].max(axis=1).astype(int)

        check_lmarks = np.logical_and((min_x != max_x), (min_y != max_y))

        correct_nums = np.where(check_lmarks)[0]

        return correct_nums


    # get index for target and reference
    def get_image_index(self, n_frames_total, cur_seq_len, max_t_step=4):            
        n_frames_total = min(cur_seq_len, n_frames_total)             # total number of frames to load
        max_t_step = min(max_t_step, (cur_seq_len-1) // max(1, (n_frames_total-1)))        
        t_step = np.random.randint(max_t_step) + 1                    # spacing between neighboring sampled frames                
        
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible frame index for the first frame

        start_idx = np.random.randint(offset_max)                 # offset for the first frame to load    

        # indices for target
        target_ids = [start_idx + step * t_step for step in range(self.n_frames_total)]

        # indices for reference frames
        if self.opt.isTrain:
            max_range, min_range = 300, 14                            # range for possible reference frames
            ref_range = list(range(max(0, start_idx - max_range), max(1, start_idx - min_range))) \
                    + list(range(min(start_idx + min_range, cur_seq_len - 1), min(start_idx + max_range, cur_seq_len)))
            ref_indices = np.random.choice(ref_range, size=self.num_frames)   
        else:
            ref_indices = self.opt.ref_img_id.split(',')
            ref_indices = [int(i) for i in ref_indices]

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
                    drawEdge(im_edges, curve_x, curve_y, bw=bw)        
        input_tensor = transform_L(Image.fromarray(im_edges))
        return input_tensor

    # preprocess for landmarks
    def get_keypoints(self, keypoints, transform_L, size, bw=1):
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
    def get_image(self, image, transform_I):
        # transform
        img = mmcv.bgr2rgb(image)
        crop_size = Image.fromarray(img).size

        # debug
        img = transform_I(Image.fromarray(img))

        return img, crop_size

    # get image and landmarks
    def prepare_datas(self, images, lmarks, choice_ids):
        # get images and landmarks
        result_lmarks = []
        result_images = []
        for choice in choice_ids:
            image, crop_size = self.get_image(images[choice], self.transform)
            lmark = self.get_keypoints(lmarks[choice], self.transform_L, crop_size)

            result_lmarks.append(lmark)
            result_images.append(image)

        return result_images, result_lmarks

    def reverse_rt(self, source, RT):
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

    def get_warp_ref(self, tgt_rt, ref_rt, ref_indexs, target_id):
        warp_ref_ids = []
        
        for gg in target_id:
            # select
            reference_rt_diffs = []
            target_rt = tgt_rt[gg]
            for t in ref_indexs:
                reference_rt_diffs.append(ref_rt[t] - target_rt)
            # most similar reference to target
            reference_rt_diffs = np.mean(np.absolute(reference_rt_diffs), axis=1)
            warp_ref_ids.append(np.argmin(reference_rt_diffs))

        return warp_ref_ids