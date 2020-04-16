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
from scipy.io import wavfile
import copy

from data.base_dataset import BaseDataset, get_transform
from data.keypoint2img import interpPoints, drawEdge
from util.util import openrate
from util.util import get_roi, get_roi_backup, eye_blinking
from util import face_utils

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
        parser.add_argument('--tgt_audio_path', type=str, default=None)
        parser.add_argument('--ref_video_path', type=str)
        parser.add_argument('--ref_lmarks_path', type=str)
        parser.add_argument('--ref_rt_path', type=str, default=None)
        parser.add_argument('--ref_front_path', type=str, default=None)
        parser.add_argument('--ref_ani_id', type=int)
        parser.add_argument('--ref_ani_path', type=str, default=None)
        parser.add_argument('--find_largest_mouth', action='store_true', help='find reference image that open mouth in largest ratio')
        parser.add_argument('--crop_ref', action='store_true')
        parser.add_argument('--no_head_motion', action='store_true')
        parser.add_argument('--origin_not_require', action='store_true')
        parser.add_argument('--audio_append', type=int, default=1, help='number of chunck to append')

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

        if self.opt.audio_drive:
            self.tgt_part_list = copy.deepcopy(self.part_list)
            self.part_list = [[list(range(0, 17)) + ((list(range(68, 83)) + [0]) if self.add_upper_face else [])], # face
                        [range(17, 22)],                                  # right eyebrow
                        [range(22, 27)],                                  # left eyebrow
                        [[28, 31], range(31, 36), [35, 28]],              # nose
                        [[36,37,38,39], [39,40,41,36]],                   # right eye
                        [[42,43,44,45], [45,46,47,42]],                   # left eye
                    ]
       
        # load path
        self.tgt_video_path = opt.tgt_video_path
        self.tgt_lmarks_path = opt.tgt_lmarks_path
        self.tgt_ani_path = opt.tgt_ani_path
        self.tgt_rt_path = opt.tgt_rt_path
        if self.opt.audio_drive:
            self.tgt_audio_path = opt.tgt_audio_path

        self.ref_video_path = opt.ref_video_path
        self.ref_lmarks_path = opt.ref_lmarks_path
        self.ref_rt_path = opt.ref_rt_path
        self.ref_front_path = opt.ref_front_path

        # read in data
        self.tgt_lmarks = np.load(self.tgt_lmarks_path) #[:,:,:-1]
        self.tgt_video = self.read_videos(self.tgt_video_path)
        if self.opt.audio_drive:
            fs, self.tgt_audio = wavfile.read(self.tgt_audio_path)
            self.chunck_size = int(fs/25)

        # get enough video associate with landmark
        if len(self.tgt_video) < self.tgt_lmarks.shape[0]:
            self.tgt_video.extend([self.tgt_video[-1] for i in range(self.tgt_lmarks.shape[0]-len(self.tgt_video))])

        self.ref_lmarks = np.load(self.ref_lmarks_path)
        self.ref_video = self.read_videos(self.ref_video_path)
        # pdb.set_trace()
        if self.opt.warp_ani:
            self.tgt_ani_video = self.read_videos(self.tgt_ani_path)
            self.ref_front = np.load(self.ref_front_path)
            self.ref_ani_id = self.opt.ref_ani_id
        if self.opt.warp_ani or self.ref_search:
            self.tgt_rt = np.load(self.tgt_rt_path)
            self.ref_rt = np.load(self.ref_rt_path)

        # clean
        correct_nums, wro_nums = self.clean_lmarks(self.tgt_lmarks)
        self.tgt_lmarks = self.tgt_lmarks[correct_nums]
        self.tgt_video = np.asarray(self.tgt_video)[correct_nums]
        if self.opt.warp_ani:
            self.tgt_ani_video = np.asarray(self.tgt_ani_video)[correct_nums]
        if self.opt.warp_ani or self.ref_search:
            self.tgt_rt = self.tgt_rt[correct_nums]
        # audio
        if self.opt.audio_drive:
            if wro_nums.shape[0] != 0:
                self.tgt_audio = self.clean_audio(self.chunck_size, self.tgt_audio, wro_nums)

            # append
            left_append = self.tgt_audio[:self.opt.audio_append*self.chunck_size]
            right_append = self.tgt_audio[-(self.opt.audio_append+1)*self.chunck_size:]
            self.tgt_audio = np.insert(self.tgt_audio, 0, left_append, axis=0)
            self.tgt_audio = np.insert(self.tgt_audio, -1, right_append, axis=0)

        # smooth landmarks
        for i in range(self.tgt_lmarks.shape[1]):
            x = self.tgt_lmarks[:, i, 0]
            x = face_utils.smooth(x, window_len=5)
            self.tgt_lmarks[: ,i, 0] = x[2:-2]
            y = self.tgt_lmarks[:, i, 1]
            y = face_utils.smooth(y, window_len=5)
            self.tgt_lmarks[: ,i, 1] = y[2:-2]

        # get eyes
        # self.tgt_lmarks = eye_blinking(self.tgt_lmarks)

        correct_nums,_ = self.clean_lmarks(self.ref_lmarks)
        self.ref_lmarks = self.ref_lmarks[correct_nums]
        self.ref_video = np.asarray(self.ref_video)[correct_nums]
        if self.opt.warp_ani or self.ref_search:
            self.ref_rt = self.ref_rt[correct_nums]

        # smooth landmarks
        for i in range(self.ref_lmarks.shape[1]):
            x = self.ref_lmarks[:, i, 0]
            x = face_utils.smooth(x, window_len=5)
            self.ref_lmarks[: ,i, 0] = x[2:-2]
            y = self.ref_lmarks[:, i, 1]
            y = face_utils.smooth(y, window_len=5)
            self.ref_lmarks[: ,i, 1] = y[2:-2]

        # get transform for image and landmark
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self.__scale_image(img, self.output_shape, Image.BICUBIC)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.transform_L = transforms.Compose([
            transforms.Lambda(lambda img: self.__scale_image(img, self.output_shape, Image.BILINEAR)),
            transforms.ToTensor()
        ])

        self.transform_T = transforms.Compose([
            transforms.Lambda(lambda img: self.__scale_image(img, self.output_shape, Image.BILINEAR)),
        ])

        # define parameters for inference
        self.ref_lmarks_temp = self.ref_lmarks
        self.ref_video, self.ref_lmarks, self.ref_indices = self.define_inference(self.ref_video, self.ref_lmarks)

        # reference animation
        # if self.ref_ani_path[-3:] == 'mp4':
        #     self.ref_ani_video = 

        if self.opt.dataset_name == 'lrw':
            for img_id in range(self.ref_video.shape[0]):
                mask = self.ref_video[img_id] == 1
                self.ref_video[img_id] = self.ref_video[img_id] - 2*mask.type(torch.uint8)


    def __len__(self):
        return len(self.tgt_video) 

    def __scale_image(self, img, size, method=Image.BICUBIC):
        w, h = size    
        return img.resize((w, h), method)

    def __color_aug(self, img, params):
        h, s, v = img.convert('HSV').split()    
        h = h.point(lambda i: (i + params[0]) % 256)
        s = s.point(lambda i: min(255, max(0, i * params[1] + params[2])))
        v = v.point(lambda i: min(255, max(0, i * params[3] + params[4])))
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
        return img

    def name(self):
        return 'FaceForensicsDemoDataset'

    def __getitem__(self, index):
        # get target
        target_id = [index]
        tgt_images, tgt_lmarks = self.prepare_datas(self.tgt_video, self.tgt_lmarks, target_id)

        # get audio for reference and target
        if self.opt.audio_drive:
            tgt_mfcc = []
            for ind in target_id:
                tgt_mfcc.append(\
                    self.tgt_audio[ind*self.chunck_size:\
                        (ind+2*self.opt.audio_append+1)*self.chunck_size])

        # get animation
        if self.opt.warp_ani:
            # get animation & get cropped ground truth
            ani_lmarks = []
            ani_images = []
            ani_lmarks_temp = []

            for gg in target_id:
                if self.opt.no_head_motion:
                    ani_lmarks.append(self.ref_front[int(self.ref_ani_id)])
                else:
                    ani_lmarks.append(self.reverse_rt(self.ref_front[int(self.ref_ani_id)], self.tgt_rt[gg]))
                ani_lmarks[-1] = np.array(ani_lmarks[-1])
                ani_lmarks_temp.append(ani_lmarks[-1])
                if self.opt.no_head_motion:
                    ani_images.append(self.tgt_ani_video[self.ref_ani_id])
                else:
                    ani_images.append(self.tgt_ani_video[gg])

            ani_images, ani_lmarks = self.prepare_datas(ani_images, ani_lmarks, list(range(len(target_id))))

            # crop by mask
            if self.opt.crop_ref:
                for ani_lmark_id, ani_lmark_temp in enumerate(ani_lmarks_temp):
                    ani_template = torch.Tensor(self.get_template(ani_lmark_temp, self.transform_T))
                    ani_template_inter = -ani_images[ani_lmark_id] * ani_template + (1 - ani_template)
                    ani_images[ani_lmark_id] = ani_images[ani_lmark_id] / ani_template_inter

            # finetune (double check)
            if self.opt.finetune and self.opt.origin_not_require:
                ori_ani_image, ori_ani_lmark, ori_ani_lmarks_temp = [], [], []
                for ref_idx in self.ref_indices:
                    ori_ani_lmark.append(self.reverse_rt(self.ref_front[int(self.ref_ani_id)], self.tgt_rt[ref_idx]))
                    ori_ani_lmark[-1] = np.array(ori_ani_lmark[-1])
                    ori_ani_lmarks_temp.append(ori_ani_lmark[-1])
                    ori_ani_image.append(self.tgt_ani_video[ref_idx])

                ori_ani_image, ori_ani_lmark = self.prepare_datas(ori_ani_image, ori_ani_lmark, list(range(len(self.ref_indices))))

                # crop by mask
                if self.opt.crop_ref:
                    for ani_lmark_id, ani_lmark_temp in enumerate(ori_ani_lmarks_temp):
                        ani_template = torch.Tensor(self.get_template(ani_lmark_temp, self.transform_T))
                        ani_template_inter = -ori_ani_image[ani_lmark_id] * ani_template + (1 - ani_template)
                        ori_ani_image[ani_lmark_id] = ori_ani_image[ani_lmark_id] / ani_template_inter

            # reference animation
            elif self.opt.finetune and not self.opt.origin_not_require:
                ori_ani_image = [cv2.imread(self.opt.ref_ani_path)]
                ori_ani_lmark_temp = self.ref_lmarks
                ori_ani_image, ori_ani_lmark = self.prepare_datas(ori_ani_image, ori_ani_lmark_temp, [0])

                # crop by mask
                if self.opt.crop_ref:
                    for ani_lmark_id, ani_lmark_temp in enumerate(self.ref_lmarks_temp):
                        ani_template = torch.Tensor(self.get_template(ani_lmark_temp, self.transform_T))
                        ani_template_inter = -ori_ani_image[ani_lmark_id] * ani_template + (1 - ani_template)
                        ori_ani_image[ani_lmark_id] = ori_ani_image[ani_lmark_id] / ani_template_inter

        # get warping reference
        if self.ref_search and not self.opt.no_head_motion:
            ref_rt = self.ref_rt[:, :3]
            tgt_rt = self.tgt_rt[:, :3]
            warping_ref_ids = self.get_warp_ref(tgt_rt, ref_rt, self.ref_indices, target_id)
            warping_refs = [self.ref_video[w_ref_id] for w_ref_id in warping_ref_ids]
            warping_ref_lmarks = [self.ref_lmarks[w_ref_id] for w_ref_id in warping_ref_ids]
        elif self.opt.no_head_motion:
            ref_rt = self.ref_rt[:, :3]
            warping_ref_ids = self.get_warp_ref(ref_rt, ref_rt, self.ref_indices, [int(self.ref_ani_id)])
            warping_refs = [self.ref_video[w_ref_id] for w_ref_id in warping_ref_ids]
            warping_ref_lmarks = [self.ref_lmarks[w_ref_id] for w_ref_id in warping_ref_ids]
        else:
            warping_refs = [self.ref_video[0] for t_id in target_id]
            warping_ref_lmarks = [self.ref_lmarks[0] for t_id in target_id]

        # crop by mask
        if self.opt.crop_ref:
            for warp_id, warp_ref in enumerate(warping_refs):
                lmark_id = self.ref_indices[warping_ref_ids[warp_id]]
                warp_ref_lmark = self.ref_lmarks_temp[lmark_id]
                warp_ref_template = torch.Tensor(self.get_template(warp_ref_lmark, self.transform_T))
                warp_ref_template_inter = -warp_ref * warp_ref_template + (1 - warp_ref_template)
                warping_refs[warp_id] = warp_ref / warp_ref_template_inter

        target_img_path = [os.path.join(self.tgt_video_path[:-4] , '%05d.png'%t_id) for t_id in target_id]

        # audio
        if self.opt.audio_drive:
            tgt_mfcc = torch.cat([torch.Tensor(chunck).unsqueeze(0).unsqueeze(0) for chunck in tgt_mfcc])
            # get gt landmark
            tmp_list = self.part_list
            self.part_list = self.tgt_part_list
            _, tgt_gt_lmarks = self.prepare_datas(self.tgt_video, self.tgt_lmarks, target_id)
            self.part_list = tmp_list
            tgt_gt_lmarks = torch.cat([tgt_gt_lmark.unsqueeze(0) for tgt_gt_lmark in tgt_gt_lmarks], axis=0)

        tgt_images = torch.cat([tgt_img.unsqueeze(0) for tgt_img in tgt_images], axis=0)
        tgt_lmarks = torch.cat([tgt_lmark.unsqueeze(0) for tgt_lmark in tgt_lmarks], axis=0)
        if self.opt.dataset_name == 'lrw':
            mask = tgt_images == 1
            tgt_images = tgt_images - 2 * mask.type(torch.uint8)
        if self.opt.warp_ani:
            ani_images = torch.cat([ani_image.unsqueeze(0) for ani_image in ani_images], axis=0)
            ani_lmarks = torch.cat([ani_lmark.unsqueeze(0) for ani_lmark in ani_lmarks], axis=0)
            if self.opt.finetune:
                ori_ani_image = torch.cat([ori_im.unsqueeze(0) for ori_im in ori_ani_image], axis=0)
                ori_ani_lmark = torch.cat([ori_lm.unsqueeze(0) for ori_lm in ori_ani_lmark], axis=0)
        # if self.ref_search:
        warping_refs = torch.cat([warping_ref.unsqueeze(0) for warping_ref in warping_refs], axis=0)
        warping_ref_lmarks = torch.cat([warping_ref_lmark.unsqueeze(0) for warping_ref_lmark in warping_ref_lmarks], axis=0)

        input_dic = {'path': self.tgt_video_path, 'v_id' : target_img_path, 'index':target_id, 'tgt_label': tgt_lmarks, \
            'ref_image':self.ref_video , 'ref_label': self.ref_lmarks, \
            'tgt_image': tgt_images,  'target_id': target_id}
        if self.opt.warp_ani:
            input_dic.update({'ani_image': ani_images, 'ani_lmark': ani_lmarks})
            if self.opt.finetune:
                input_dic.update({'ori_ani_image': ori_ani_image, 'ori_ani_lmark': ori_ani_lmark})
        # if self.ref_search:
        input_dic.update({'warping_ref': warping_refs, 'warping_ref_lmark': warping_ref_lmarks})
        if self.opt.audio_drive:
            input_dic.update({'tgt_audio': tgt_mfcc, 'tgt_gt_label': tgt_gt_lmarks})

        return input_dic

    # define parameters for inference
    def define_inference(self, real_video, lmarks):
        # get reference index (not only for one shot)
        if self.opt.find_largest_mouth:
            openrates = []
            for i in range(len(lmarks)):
                openrates.append(openrate(lmarks[i]))
            openrates = np.asarray(openrates)
            max_index = np.argsort(openrates)
            ref_indices = max_index[:self.opt.n_shot]
        else:
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
        wrong_nums = np.where(check_lmarks!=True)[0]

        return correct_nums, wrong_nums

    # clean mfcc base on nums
    def clean_audio(self, chunck_size, mfcc, wro_nums):
        delete_indices = []
        for num in wro_nums:
            delete_indices += list(range(num*chunck_size, (num+1)*chunck_size))
        return np.delete(mfcc, delete_indices, axis=0)

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
            # lmark = self.get_keypoints(lmarks[choice].copy(), self.transform_L, crop_size)
            # get landmark
            count = 0
            while True:
                try:
                    lmark = self.get_keypoints(lmarks[choice].copy(), self.transform_L, crop_size)
                    break
                except:
                    choice = ((choice + 1)%images.shape[0])
                    print("what the fuck for {}".format(self.tgt_video_path))
                    count += 1
                    if count > 20:
                        break

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
            if self.opt.no_head_motion:
                warp_ref_ids.append(random.randint(0, len(ref_indexs)-1))
            else:
                warp_ref_ids.append(np.argmin(reference_rt_diffs))
            # warp_ref_ids.append(np.argmax(reference_rt_diffs))

        return warp_ref_ids

    # preprocess for template
    def get_template(self, lmark, transform_T):
        # crop
        if self.opt.dataset_name == 'grid':
            template = get_roi_backup(lmark)
        else:
            template = get_roi(lmark)
        template = Image.fromarray(template[:, :, 0], 'L')
        template = np.asarray(transform_T(template))
        
        return template