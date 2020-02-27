# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import numpy as np
import torch
import cv2
from collections import OrderedDict
from PIL import Image
import pickle as pkl

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html

import mmcv
from tqdm import tqdm
import pdb

import warnings
warnings.simplefilter('ignore')

def get_param(root, pickle_data, pick_id, opt):
    paths = pickle_data[pick_id]
    if opt.dataset_name == 'vox':
        # target
        opt.tgt_video_path = os.path.join(root, 'unzip/test_video', paths[0], paths[1], paths[2]+"_aligned.mp4")
        opt.tgt_lmarks_path = os.path.join(root, 'unzip/test_video', paths[0], paths[1], paths[2]+"_aligned.npy")
        opt.tgt_rt_path = os.path.join(root, 'unzip/test_video', paths[0], paths[1], paths[2]+"_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, 'unzip/test_video', paths[0], paths[1], paths[2]+"_aligned_ani.mp4")
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, 'unzip/test_video', ref_paths[0], ref_paths[1], ref_paths[2]+"_aligned_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = int(ref_paths[3])

        audio_tgt_path = os.path.join(root, 'unzip/test_audio', paths[0], paths[1], paths[2]+".m4a")

    elif opt.dataset_name == 'grid':
        # target
        opt.tgt_video_path = os.path.join(root, 'align', paths[0], paths[1]+"_crop.mp4")
        opt.tgt_lmarks_path = os.path.join(root, 'align', paths[0], paths[1]+"_original.npy")
        opt.tgt_rt_path = None
        opt.tgt_ani_path = None
        # reference
        ref_paths = paths
        opt.ref_front_path = None
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = None

        audio_tgt_path = os.path.join(root, 'audio', paths[0], paths[1]+".wav")

    elif opt.dataset_name == 'lrs':
        # target
        paths[1] = paths[1].split('_')[0]
        opt.tgt_video_path = os.path.join(root, 'test', paths[0], paths[1]+"_crop.mp4")
        opt.tgt_lmarks_path = os.path.join(root, 'test', paths[0], paths[1]+"_original.npy")
        opt.tgt_rt_path = os.path.join(root, 'test', paths[0], paths[1]+"_rt.npy")
        opt.tgt_ani_path = os.path.join(root, 'test', paths[0], paths[1]+"_ani.mp4")
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, 'test', paths[0], paths[1]+"_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = int(paths[2])

        audio_tgt_path = os.path.join(root, 'test', paths[0], paths[1]+".wav")

    elif opt.dataset_name == 'lrw':
        # target
        opt.tgt_video_path = os.path.join(paths[0]+"_crop.mp4")
        opt.tgt_lmarks_path = os.path.join(paths[0]+"_original.npy")
        opt.tgt_rt_path = os.path.join(paths[0]+"_rt.npy")
        opt.tgt_ani_path = os.path.join(paths[0]+"_ani.mp4")
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(ref_paths[0]+"_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = int(ref_paths[1])

        audio_tgt_path = os.path.join(paths[0].replace('video', 'audio')+".wav")

    return audio_tgt_path

opt = TestOptions().parse()

### setup models
model = create_model(opt)
model.eval()

root = opt.dataroot
if opt.dataset_name == 'grid':
    _file = open(os.path.join(root, 'pickle','test_audio2lmark_grid.pkl'), "rb")
elif opt.dataset_name == 'lrw':
    _file = open(os.path.join(root, 'pickle','test3_lmark2img.pkl'), "rb")
else:
    _file = open(os.path.join(root, 'pickle','test_lmark2img.pkl'), "rb")
pickle_data = pkl.load(_file)
_file.close()

# save_root = os.path.join('/home/cxu-serve/p1/common/grid_gen')
save_root = os.path.join('/home/cxu-serve/p1/common/lrw_gen')
start_id = len(pickle_data) // 3 * 2
# end_id = len(pickle_data) // 3 * 2
# end_id = len(pickle_data)
# start_id = 0
end_id = len(pickle_data)
pick_ids = range(start_id, end_id)

for pick_id in tqdm(pick_ids):
    print(pick_id)
    paths = pickle_data[pick_id]
    audio_tgt_path = get_param(root, pickle_data, pick_id, opt)

    ### setup dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # test
    # ref_idx_fix = torch.zeros([opt.batchSize])
    ref_idx_fix = None
    frames = []
    for i, data in enumerate(dataset):
        if i >= len(dataset): break
        img_path = data['path']
        if not opt.warp_ani:
            data.update({'ani_image':None, 'ani_lmark':None, 'cropped_images':None, 'cropped_lmarks':None })
        if "warping_ref" not in data:
            data.update({'warping_ref': data['ref_image'][:, :1], 'warping_ref_lmark': data['ref_label'][:, :1]})

        img_path = data['path']
        data_list = [data['tgt_label'], data['tgt_image'], None, None, None, None, \
                    data['ref_label'], data['ref_image'], \
                    data['warping_ref_lmark'].squeeze(1) if data['warping_ref_lmark'] is not None else None, \
                    data['warping_ref'].squeeze(1) if data['warping_ref'] is not None else None, \
                    data['ani_lmark'].squeeze(1) if opt.warp_ani else None, \
                    data['ani_image'].squeeze(1) if opt.warp_ani else None, \
                    None, None, None]
        synthesized_image, _, _, _, _, _, _, _, _, _ = model(data_list, ref_idx_fix=ref_idx_fix)
        
        for batch in range(synthesized_image.shape[0]):
            img = util.tensor2im(synthesized_image[batch])
            frames.append(np.expand_dims(img, axis=0))

    # for f_id, f in enumerate(frames):
    #     img = Image.fromarray(f[0])
    #     img.save('example/lrw/{}.png'.format(f_id))
    frames_for_save = np.concatenate(frames, axis=0)
    # save image
    if opt.dataset_name == 'grid':
        img_dir = os.path.join(save_root,  paths[0])
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_name = os.path.join(img_dir, "{}.npy".format(paths[1]))
        np.save(img_name, frames_for_save)
    elif opt.dataset_name == 'lrw':
        dir_paths = paths[0].split('/')
        img_dir = os.path.join(save_root, os.path.join(dir_paths[-3], dir_paths[-2]))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_name = os.path.join(img_dir, "{}.npy".format(dir_paths[-1]))
        np.save(img_name, frames_for_save)