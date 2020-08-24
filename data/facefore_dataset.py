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
from scipy.io import wavfile

from data.base_dataset import BaseDataset, get_transform
from data.keypoint2img import interpPoints, drawEdge
from scipy.spatial.transform import Rotation as R
from util.util import get_roi, get_roi_small_eyes, get_abso_mouth
from util import face_utils
from util.util import openrate

import copy
import pdb

class FaceForeDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        parser.add_argument('--aspect_ratio', type=float, default=1)        
        parser.add_argument('--no_upper_face', action='store_true', help='do not add upper face')
        parser.add_argument('--use_ft', action='store_true')
        parser.add_argument('--dataset_name', type=str, help='face or vox or grid')
        parser.add_argument('--worst_ref_prob', type=float, default=0.75, help='probability to select worst reference image')
        parser.add_argument('--ref_ratio', type=float, default=0.25, help='ratio to select reference images')
        parser.add_argument('--crop_ref', action='store_true')
        parser.add_argument('--find_largest_mouth', action='store_true', help='find reference image that open mouth in largest ratio')
        parser.add_argument('--audio_append', type=int, default=1, help='number of chunck to append')

        # for reference
        parser.add_argument('--ref_img_id', type=str)
        parser.add_argument('--no_head_motion', action='store_true')
        parser.add_argument('--for_finetune', action='store_true')

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
        self.n_frames_total = 1
        self.opt = opt
        self.root  = opt.dataroot
        self.fix_crop_pos = False
        self.use_for_finetune = False  #(whether for finetune)
        # self.fix_crop_pos = False

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
        
        self.load_pickle(opt)

        # self.data = self.data[:2]
        print(len(self.data))

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
        if self.opt.dataset_name == 'face':
            video_path = self.data[index][1] #os.path.join(self.root, 'pretrain', v_id[0] , v_id[1][:5] + '_crop.mp4'  )
            lmark_path = self.data[index][0]  #= os.path.join(self.root, 'pretrain', v_id[0] , v_id[1]  )
        elif self.opt.dataset_name == 'vox':
            paths = self.data[index]
            video_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned.mp4")
            if self.opt.no_head_motion:
                lmark_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned_front.npy")
            else:
                lmark_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned.npy")
            ani_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned_ani.mp4")
            rt_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned_rt.npy")
            front_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned_front.npy")
            ani_id = paths[3]

        elif self.opt.dataset_name == 'grid':
            paths = self.data[index]
            video_path = os.path.join(self.root, self.video_bag, paths[0], paths[1] + '_crop.mp4')
            lmark_path = os.path.join(self.root, self.video_bag, paths[0], paths[1]+ '_original.npy') 
            rt_path = os.path.join(self.root, self.video_bag, paths[0], paths[1]+ '_rt.npy') 
            front_path = os.path.join(self.root, self.video_bag, paths[0], paths[1]+ '_front.npy') 

        elif self.opt.dataset_name == 'lrs':
            paths = self.data[index]
            paths[1] = paths[1].split('_')[0]
            video_path = os.path.join(self.root, self.video_bag, paths[0], paths[1] + '_crop.mp4')
            lmark_path = os.path.join(self.root, self.video_bag, paths[0], paths[1]+ '_original.npy') 
            ani_path = os.path.join(self.root, self.video_bag, paths[0], paths[1]+"_ani.mp4")
            rt_path = os.path.join(self.root, self.video_bag, paths[0], paths[1]+ '_rt.npy') 
            front_path = os.path.join(self.root, self.video_bag, paths[0], paths[1]+ '_front.npy')
            if self.opt.warp_ani:
                ani_id = int(paths[2])

        elif self.opt.dataset_name == 'lrw':
            paths = self.data[index]
            video_path = os.path.join(paths[0] + '_crop.mp4')
            lmark_path = os.path.join(paths[0]+ '_original.npy') 
            ani_path = os.path.join(paths[0]+"_ani.mp4")
            rt_path = os.path.join(paths[0]+ '_rt.npy') 
            front_path = os.path.join(paths[0]+ '_front.npy')
            if self.opt.warp_ani:
                ani_id = int(paths[1])

        elif self.opt.dataset_name == 'crema':
            paths = self.data[index]
            video_path = os.path.join(self.root, self.video_bag, paths[0][:-10] + '_crop.mp4')
            lmark_path = os.path.join(self.root, self.video_bag, paths[0][:-10] + '_original.npy') 
            rt_path = os.path.join(self.root, self.video_bag, paths[0][:-10] + '_rt.npy') 
            front_path = os.path.join(self.root, self.video_bag, paths[0][:-10] + '_front.npy') 
        
        elif self.opt.dataset_name == 'obama':
            paths = self.data[index]
            video_path = os.path.join(self.root, self.video_bag, paths[0][:-11] + '_crop2.mp4')
            lmark_path = os.path.join(self.root, self.video_bag, paths[0][:-11] + '_original2.npy')
            ani_path = os.path.join(self.root, self.video_bag, paths[0][:-11]+"_ani2.mp4")
            rt_path = os.path.join(self.root, self.video_bag, paths[0][:-11] + '_rt2.npy') 
            front_path = os.path.join(self.root, self.video_bag, paths[0][:-11] + '_front2.npy') 

            ani_id = int(paths[1])

        elif self.opt.dataset_name == 'ouyang':
            self.real_video = None
            self.ani_video = None
            self.use_for_finetune = True

            paths = self.data[index]
            video_path = os.path.join(self.root, self.video_bag, paths + '_crop.mp4')
            lmark_path = os.path.join(self.root, self.video_bag, paths + '__original.npy')
            ani_path = os.path.join(self.root, self.video_bag, paths+"__ani.mp4")
            rt_path = os.path.join(self.root, self.video_bag, paths + '__rt.npy') 
            front_path = os.path.join(self.root, self.video_bag, paths + '__front.npy') 

            ani_id = 11174

        # reseed
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        # read in data
        self.video_path = video_path
        lmarks = np.load(lmark_path)#[:,:,:-1]

        real_video = self.read_videos(video_path)
        
        if self.opt.dataset_name == 'face':
            lmarks = lmarks[:-1]
        else:
            if self.opt.warp_ani:
                front = np.load(front_path)
            rt = np.load(rt_path)
        cor_num, wro_nums = self.clean_lmarks(lmarks)
        lmarks = lmarks[cor_num]
        real_video = np.asarray(real_video)[cor_num]
        rt = rt[cor_num]

        # smooth landmarks
        for i in range(lmarks.shape[1]):
            x = lmarks[:, i, 0]
            x = face_utils.smooth(x, window_len=5)
            lmarks[: ,i, 0] = x[2:-2]
            y = lmarks[:, i, 1]
            y = face_utils.smooth(y, window_len=5)
            lmarks[: ,i, 1] = y[2:-2]

        if self.opt.warp_ani:
            ani_video = self.read_videos(ani_path)
            # clean data
            ani_video = np.asarray(ani_video)[cor_num]

        v_length = len(real_video)

        # sample index of frames for embedding network
        if self.opt.ref_ratio is not None:
            input_indexs, target_id = self.get_image_index_ratio(self.n_frames_total, v_length)
        elif self.opt.for_finetune:
            input_indexs, target_id = self.get_image_index_finetune(self.n_frames_total, lmarks=lmarks)
        else:
            input_indexs, target_id = self.get_image_index(self.n_frames_total, v_length)
    
        # whether get open mouth
        if self.opt.find_largest_mouth:
            result_indexs = self.get_open_mouth(lmarks)
            input_indexs = result_indexs if result_indexs is not None else input_indexs

        # define scale
        scale = self.define_scale()
        transform, transform_L, transform_T = self.get_transforms()

        # get reference
        ref_images, ref_lmarks, ref_coords = self.prepare_datas(real_video, lmarks, input_indexs, transform, transform_L, scale)

        # get target
        tgt_images, tgt_lmarks, tgt_crop_coords = self.prepare_datas(real_video, lmarks, target_id, transform, transform_L, scale)


        # get template for target
        tgt_templates = []
        tgt_templates_eyes = []
        tgt_templates_mouth = []
        for gg in target_id:
            lmark = lmarks[gg]
            tgt_templates.append(self.get_template(lmark, transform_T, self.output_shape, tgt_crop_coords))
            tgt_templates_eyes.append(self.get_template(lmark, transform_T, self.output_shape, tgt_crop_coords, only_eyes=True))
            tgt_templates_mouth.append(self.get_template(lmark, transform_T, self.output_shape, tgt_crop_coords, only_mouth=True))

        if self.opt.warp_ani:
        # get animation & get cropped ground truth
            ani_lmarks_back = []
            ani_lmarks = []
            ani_images = []
            cropped_images  = []
            cropped_lmarks = []

            for gg in target_id:
                cropped_gt = real_video[gg].copy()
                ani_lmarks.append(self.reverse_rt(front[int(ani_id)], rt[gg]))
                ani_lmarks[-1] = np.array(ani_lmarks[-1])
                ani_lmarks_back.append(ani_lmarks[-1])
                ani_images.append(ani_video[gg])
                mask = ani_video[gg] < 10
                # mask = scipy.ndimage.morphology.binary_dilation(mask.numpy(),iterations = 5).astype(np.bool)
                cropped_gt[mask] = 0
                cropped_images.append(cropped_gt)
                cropped_lmarks.append(lmarks[gg])
            ani_images, ani_lmarks, ani_coords = self.prepare_datas(ani_images, ani_lmarks, list(range(len(target_id))), transform, transform_L, scale)
            cropped_images, cropped_lmarks, _ = self.prepare_datas(cropped_images, cropped_lmarks, list(range(len(target_id))), \
                                                                    transform, transform_L, scale, crop_coords=tgt_crop_coords)
        
        # get warping reference
        rt = rt[:, :3]
        warping_ref_ids = self.get_warp_ref(rt, input_indexs, target_id)
        warping_refs = [ref_images[w_ref_id] for w_ref_id in warping_ref_ids]
        warping_ref_lmarks = [ref_lmarks[w_ref_id] for w_ref_id in warping_ref_ids]
        ori_warping_refs = copy.deepcopy(warping_refs)
        
        # get template for warp reference and animation
        if self.opt.crop_ref:
            # for warp reference
            for warp_id, warp_ref in enumerate(warping_refs):
                lmark_id = input_indexs[warping_ref_ids[warp_id]]
                warp_ref_lmark = lmarks[lmark_id]
                warp_ref_template = torch.Tensor(self.get_template(warp_ref_lmark, transform_T, self.output_shape, ref_coords))
                warp_ref_template_inter = -warp_ref * warp_ref_template + (1 - warp_ref_template)
                warping_refs[warp_id] = warp_ref / warp_ref_template_inter
            # for animation
            if self.opt.warp_ani:
                for ani_lmark_id, ani_lmark_temp in enumerate(ani_lmarks_back):
                    ani_template = torch.Tensor(self.get_template(ani_lmark_temp, transform_T, self.output_shape, ani_coords))
                    ani_template_inter = -ani_images[ani_lmark_id] * ani_template + (1 - ani_template)
                    ani_images[ani_lmark_id] = ani_images[ani_lmark_id] / ani_template_inter
            
        # preprocess
        target_img_path  = [os.path.join(video_path[:-4] , '%05d.png'%t_id) for t_id in target_id]

        ref_images = torch.cat([ref_img.unsqueeze(0) for ref_img in ref_images], axis=0)
        ref_lmarks = torch.cat([ref_lmark.unsqueeze(0) for ref_lmark in ref_lmarks], axis=0)
        tgt_images = torch.cat([tgt_img.unsqueeze(0) for tgt_img in tgt_images], axis=0)
        tgt_lmarks = torch.cat([tgt_lmark.unsqueeze(0) for tgt_lmark in tgt_lmarks], axis=0)
        if self.opt.isTrain:
            tgt_templates = torch.cat([torch.Tensor(tgt_template).unsqueeze(0).unsqueeze(0) for tgt_template in tgt_templates], axis=0)
            tgt_templates_eyes = torch.cat([torch.Tensor(tgt_template).unsqueeze(0).unsqueeze(0) for tgt_template in tgt_templates_eyes], axis=0)
            tgt_templates_mouth = torch.cat([torch.Tensor(tgt_template).unsqueeze(0).unsqueeze(0) for tgt_template in tgt_templates_mouth], axis=0)
        else:
            tgt_templates = torch.cat([torch.Tensor(tgt_template).unsqueeze(0).unsqueeze(0) for tgt_template in tgt_templates], axis=0)
            tgt_templates_eyes = torch.cat([torch.Tensor(tgt_template).unsqueeze(0).unsqueeze(0) for tgt_template in tgt_templates_eyes], axis=0)
            tgt_templates_mouth = torch.cat([torch.Tensor(tgt_template).unsqueeze(0).unsqueeze(0) for tgt_template in tgt_templates_mouth], axis=0)

        warping_refs = torch.cat([warping_ref.unsqueeze(0) for warping_ref in warping_refs], 0)
        warping_ref_lmarks = torch.cat([warping_ref_lmark.unsqueeze(0) for warping_ref_lmark in warping_ref_lmarks], 0)
        ori_warping_refs = torch.cat([ori_warping_ref.unsqueeze(0) for ori_warping_ref in ori_warping_refs], 0)
        if self.opt.warp_ani:
            ani_images = torch.cat([ani_image.unsqueeze(0) for ani_image in ani_images], 0)
            ani_lmarks = torch.cat([ani_lmark.unsqueeze(0) for ani_lmark in ani_lmarks], 0)
            cropped_images = torch.cat([cropped_image.unsqueeze(0) for cropped_image in cropped_images], 0)
            cropped_lmarks = torch.cat([cropped_lmark.unsqueeze(0) for cropped_lmark in cropped_lmarks], 0)

        # crop eyes and mouth from reference 
        if self.opt.crop_ref:
            if self.opt.warp_ani:
                crop_template_inter = -cropped_images * tgt_templates + (1 - tgt_templates)
                cropped_images = cropped_images / crop_template_inter
            tgt_template_inter = -tgt_images * tgt_templates + (1 - tgt_templates)
            tgt_mask_images = tgt_images / tgt_template_inter

        input_dic = {'v_id' : target_img_path, 'tgt_label': tgt_lmarks, 'tgt_template': tgt_templates, 'ref_image':ref_images , 'ref_label': ref_lmarks, \
        'tgt_image': tgt_images,  'target_id': target_id , 'warping_ref': warping_refs , 'warping_ref_lmark': warping_ref_lmarks, \
        'ori_warping_refs': ori_warping_refs, 'path': video_path}
        if self.opt.warp_ani:
            input_dic.update({'ani_image': ani_images, 'ani_lmark': ani_lmarks, 'cropped_images': cropped_images, 'cropped_lmarks' :cropped_lmarks })
        if self.opt.crop_ref:
            input_dic.update({'tgt_mask_images': tgt_mask_images})
        else:
            input_dic.update({'tgt_mask_images': tgt_images})

        return input_dic

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

    # get index for target and reference by ratio method
    def get_image_index_ratio(self, n_frames_total, cur_seq_len, max_t_step=4):
        # check enough frames
        if n_frames_total + self.num_frames > cur_seq_len:
            assert False
        # get lengths
        target_len = max(int(cur_seq_len * (1 - self.opt.ref_ratio)), n_frames_total)
        reference_len = max(cur_seq_len - target_len, self.num_frames)
        target_len = cur_seq_len - reference_len
        
        # frame steps, target start index
        max_t_step = min(max_t_step, (target_len-1) // max(1, (n_frames_total-1)))
        t_step = np.random.randint(max_t_step) + 1
        offset_max = max(1, target_len-(n_frames_total-1)*t_step)
        start_idx = np.random.randint(offset_max) + reference_len

        # indices for target
        target_ids = [start_idx + step * t_step for step in range(self.n_frames_total)]

        # indices for reference frames
        if self.opt.isTrain:
            ref_indices = np.random.choice(reference_len, self.num_frames, replace=False)
        else:
            ref_indices = self.opt.ref_img_id.split(',')
            ref_indices = [int(i) for i in ref_indices]

        return ref_indices, target_ids

    # get index for target and reference by ratio method (only for one shot now)
    def get_image_index_finetune(self, n_frames_total, lmarks=None):
        assert n_frames_total == 1

        # get reference index (not only for one shot)
        if self.opt.find_largest_mouth:
            assert lmarks is not None
            openrates = []
            for i in range(len(lmarks)):
                openrates.append(openrate(lmarks[i]))
            openrates = np.asarray(openrates)
            max_index = np.argsort(openrates)
            ref_indices = max_index[:self.opt.n_shot]
        else:
            ref_indices = self.opt.ref_img_id.split(',')
        ref_indices = [int(i) for i in ref_indices]

        # random select reference indexs for finetune
        assert len(ref_indices) >= self.opt.finetune_shot and len(ref_indices) >= n_frames_total
        choice_num = len(ref_indices) if len(ref_indices) <= self.opt.finetune_shot+n_frames_total \
                                      else self.opt.finetune_shot+n_frames_total
        finetune_ref_idxs = np.random.choice(len(ref_indices), choice_num, replace=False)

        finetune_ref_indices = ref_indices[finetune_ref_idxs[:self.opt.finetune_shot]]
        target_ids = ref_indices[finetune_ref_idxs[-n_frames_total:]]

        return finetune_ref_indices, target_ids

    # whether get open mouth
    def get_open_mouth(self, lmarks):
        # random set number
        ref_indices = None
        if np.random.rand() >= 0.6:
            openrates = []
            for i in range(len(lmarks)):
                openrates.append(openrate(lmarks[i]))
            openrates = np.asarray(openrates)
            max_index = np.argsort(openrates)
            ref_indices = max_index[:self.opt.n_shot*8]
            sel_ids = np.random.choice(self.opt.n_shot*8, self.opt.n_shot)
            ref_indices = ref_indices[sel_ids]
        return ref_indices

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
        if self.opt.isTrain and not self.use_for_finetune:
            keypoints[:, 0] -= crop_coords[2]
            keypoints[:, 1] -= crop_coords[0]
        else:
            bw = 1

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
        if self.opt.isTrain and not self.use_for_finetune:
            img = self.crop(Image.fromarray(img), crop_coords)
        else:
            img = Image.fromarray(img)
        crop_size = img.size

        # transform
        img = transform_I(img)

        return img, crop_size

    # preprocess for template
    def get_template(self, lmark, transform_T, size, crop_coords, mask_eyes=True, mask_mouth=True, only_eyes=False, only_mouth=False):
        # crop
        if only_eyes:
            template = get_roi_small_eyes(lmark)
        elif only_mouth:
            template = get_abso_mouth(lmark)
        else:
            template = get_roi(lmark, mask_eyes, mask_mouth)
        if self.opt.isTrain and not self.use_for_finetune:
            template = self.crop(Image.fromarray(template[:, :, 0], 'L'), crop_coords)
        else:
            template = Image.fromarray(template[:, :, 0], 'L')
        template = np.asarray(transform_T(template))
        
        return template

    # get scale for random crop
    def define_scale(self, scale_max = 0.2):
        scale = [np.random.uniform(1 - scale_max, 1 + scale_max), 
                        np.random.uniform(1 - scale_max, 1 + scale_max)]    

        return scale

    # get image and landmarks
    def prepare_datas(self, images, lmarks, choice_ids, transform, transform_L, scale, crop_coords=None):
        # get cropped coordinates
        if crop_coords is None:
            crop_lmark = lmarks[choice_ids[0]]
            crop_coords = self.get_crop_coords(crop_lmark, scale)
        bw = max(1, (crop_coords[1]-crop_coords[0]) // 256)

        # get images and landmarks
        result_lmarks = []
        result_images = []
        for choice in choice_ids:
            image, crop_size = self.get_image(images[choice], transform, self.output_shape, crop_coords)
            result_images.append(image)
            if lmarks is None:
                continue
            # get landmark
            count = 0
            while True:
                try:
                    lmark = self.get_keypoints(lmarks[choice].copy(), transform_L, crop_size, crop_coords, bw)
                    break
                except:
                    choice = ((choice + 1)%images.shape[0])
                    print("what the fuck for {}".format(self.video_path))
                    count += 1
                    if count > 20:
                        break
                    
            result_lmarks.append(lmark)
            

        return result_images, result_lmarks, crop_coords

    # get crop standard from one landmark
    def get_crop_coords(self, keypoints, scale, crop_size=None):           
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

            w *= scale[0]
            h *= scale[1]
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

    def get_warp_ref(self, rt, ref_indexs, target_id):
        warp_ref_ids = []
        
        for gg in target_id:
            # random
            if self.opt.isTrain and random.random() >= self.opt.worst_ref_prob:
                warp_ref_ids.append(random.randint(0, len(ref_indexs)-1))
            # select
            else:
                reference_rt_diffs = []
                target_rt = rt[gg]
                for t in ref_indexs:
                    reference_rt_diffs.append(rt[t] - target_rt)
                # most similar reference to target
                reference_rt_diffs = np.mean(np.absolute(reference_rt_diffs), axis=1)
                if self.opt.isTrain:
                    warp_ref_ids.append(np.argmax(reference_rt_diffs))
                else:
                    # warp_ref_ids.append(np.argmin(reference_rt_diffs))
                    warp_ref_ids.append(np.argmax(reference_rt_diffs))

        return warp_ref_ids

    def get_transforms(self):
        img_params = self.get_img_params(self.output_shape)

        if self.opt.isTrain:
            transform = transforms.Compose([
                transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BICUBIC)),
                transforms.Lambda(lambda img: self.__flip(img, img_params['flip'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            transform_T = transforms.Compose([
                transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BICUBIC)),
                transforms.Lambda(lambda img: self.__flip(img, img_params['flip'])),
            ])
            transform_L = transforms.Compose([
                transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BILINEAR)),
                transforms.Lambda(lambda img: self.__flip(img, img_params['flip'])),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BICUBIC)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            transform_L = transforms.Compose([
                transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BILINEAR)),
                transforms.ToTensor()
            ])
            transform_T = transforms.Compose([
                transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BICUBIC)),
            ])
        
        
        return transform, transform_L, transform_T

    def load_pickle(self, opt):
        if self.opt.dataset_name == 'face':
            if opt.isTrain:
                _file = open(os.path.join(self.root, 'pickle','train_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()
            else :
                _file = open(os.path.join(self.root, 'pickle','test_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()

        elif self.opt.dataset_name == 'vox':
            if opt.isTrain:
                _file = open(os.path.join(self.root, 'pickle','dev_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()
            else :
                _file = open(os.path.join(self.root, 'pickle','test_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()

            if opt.isTrain:
                self.video_bag = 'unzip/dev_video'
                self.audio_bag = 'unzip/dev_audio'
            else:
                self.video_bag = 'unzip/test_video'
                self.audio_bag = 'unzip/test_audio'
        elif self.opt.dataset_name == 'grid':
            if opt.isTrain:
                _file = open(os.path.join(self.root, 'pickle', 'train_audio2lmark_grid.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()
            else :
                _file = open(os.path.join(self.root, 'pickle','test_audio2lmark_grid.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()

            if opt.isTrain:
                self.video_bag = 'align'
            else:
                self.video_bag = 'align'
        
        elif self.opt.dataset_name == 'lrs':
            if opt.isTrain:
                _file = open(os.path.join(self.root, 'pickle','new_finetune_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()
            else :
                _file = open(os.path.join(self.root, 'pickle','new_test_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()

            if opt.isTrain:
                self.video_bag = 'test'
            else:
                self.video_bag = 'test'

        elif self.opt.dataset_name == 'lrw':
            if opt.isTrain:
                _file = open(os.path.join(self.root, 'new_pickle','new_finetune_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()
            else :
                _file = open(os.path.join(self.root, 'pickle','test3_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()

            self.video_bag = ''

        elif self.opt.dataset_name == 'crema':
            if opt.isTrain:
                _file = open(os.path.join(self.root, 'pickle','train_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()
            else :
                _file = open(os.path.join(self.root, 'pickle','train_lmark2img.pkl'), "rb")
                self.data = pkl.load(_file)
                _file.close()

            if opt.isTrain:
                self.video_bag = 'VideoFlash'
                self.data = self.data[:int(0.8*len(self.data))]
            else:
                self.video_bag = 'VideoFlash'
                self.data = self.data[int(0.8*len(self.data)):]

        elif self.opt.dataset_name == 'obama':
            _file = open(os.path.join(self.root, 'pickle','train_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
            if opt.isTrain:
                self.data = [d for d in self.data if '3__' not in d[0]]
            else:
                self.data = [d for d in self.data if '3__' in d[0]]

            self.data = [d for d in self.data if ('1_1' not in d[0] and '1_2' not in d[0] and '1_3' not in d[0]) or '11_' in d[0]]
            self.video_bag = 'video'

        elif self.opt.dataset_name == 'ouyang':
            self.data = ['ouyang'] * 6
            self.video_bag = ''