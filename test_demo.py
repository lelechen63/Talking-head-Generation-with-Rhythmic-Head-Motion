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

opt = TestOptions().parse()

### setup models
model = create_model(opt)
model.eval()

root = opt.dataroot
_file = open(os.path.join(root, 'pickle','test_lmark2img.pkl'), "rb")
pickle_data = pkl.load(_file)
_file.close()

save_root = 'evaluation_store'
# pick_ids = np.random.choice(list(range(len(pickle_data))), size=opt.how_many)
pick_ids = range(0, len(pickle_data), int(len(pickle_data))//opt.how_many)

for pick_id in tqdm(pick_ids):
    print('process {} ...'.format(pick_id))

    paths = pickle_data[pick_id]
    opt.tgt_video_path = os.path.join(root, 'unzip/test_video', paths[0], paths[1], paths[2]+"_aligned.mp4")
    opt.tgt_lmarks_path = os.path.join(root, 'unzip/test_video', paths[0], paths[1], paths[2]+"_aligned.npy")
    opt.ref_video_path = opt.tgt_video_path
    opt.ref_lmarks_path = opt.tgt_lmarks_path

    ### setup dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # test
    # ref_idx_fix = torch.zeros([opt.batchSize])
    ref_idx_fix = None
    for i, data in enumerate(dataset):
        if i >= len(dataset): break
        img_path = data['path']   
        data_list = [data['tgt_label'], data['tgt_image'], None, None, data['ref_label'], data['ref_image'], None, None]
        synthesized_image, _, _, _, _, _, _, _, _ = model(data_list, ref_idx_fix=ref_idx_fix)
        
        synthesized_image = util.tensor2im(synthesized_image)    
        tgt_image = util.tensor2im(data['tgt_image'])
        tgt_lmarks = util.tensor2im(data['tgt_label'])    
        compare_image = np.hstack([tgt_lmarks, tgt_image, synthesized_image])    

        img_id = "{}_{}_{}".format(img_path[0].split('/')[-3], img_path[0].split('/')[-2], img_path[0].split('/')[-1][:-4])
        img_dir = os.path.join(save_root,  img_id)
        img_name = "%06d.jpg"%data['index'][0]

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        image_pil = Image.fromarray(compare_image)
        image_pil.save(os.path.join(img_dir, img_name))

        # save reference
        if i == 0:
            if not os.path.exists(os.path.join(img_dir, 'reference')):
                os.makedirs(os.path.join(img_dir, 'reference'))
            for ref_img_id in range(data['ref_image'].shape[1]):
                ref_img = util.tensor2im(data['ref_image'][0, ref_img_id])
                ref_img = Image.fromarray(ref_img)
                ref_img.save(os.path.join(img_dir, 'reference', 'ref_{}.jpg').format(ref_img_id))

        # save for evaluation
        if opt.evaluate:
            if not os.path.exists(os.path.join(img_dir, 'real')):
                os.makedirs(os.path.join(img_dir, 'real'))
            img_path = os.path.join(img_dir, 'real', '{}_{}_image.jpg'.format(data['index'][0], 'real'))
            image_pil = Image.fromarray(tgt_image)
            image_pil.save(img_path)

            if not os.path.exists(os.path.join(img_dir, 'synthesized')):
                os.makedirs(os.path.join(img_dir, 'synthesized'))
            img_path = os.path.join(img_dir, 'synthesized', '{}_{}_image.jpg'.format(data['index'][0], 'synthesized'))
            image_pil = Image.fromarray(synthesized_image)
            image_pil.save(img_path)

        # print('process image... %s' % img_path)

    # combine into video
    mmcv.frames2video(img_dir, os.path.join(img_dir, 'test.mp4'))