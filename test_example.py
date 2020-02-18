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

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html

import pdb

opt = TestOptions().parse()

### setup dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

### setup models
model = create_model(opt)
model.eval()
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
if opt.finetune: web_dir += '_finetune'
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch), infer=True)

# test
ref_idx_fix = None
for i, data in enumerate(dataset):
    if i >= opt.how_many or i >= len(dataset): break
    if not opt.warp_ani:
        data.update({'ani_image':None, 'ani_lmark':None, 'cropped_images':None, 'cropped_lmarks':None })

    img_path = data['path']
    data_list = [data['tgt_label'], data['tgt_image'], data['cropped_images'], None, None, \
                 data['ref_label'], data['ref_image'], data['warping_ref_lmark'].squeeze(1), data['warping_ref'].squeeze(1), \
                 data['ani_lmark'].squeeze(1) if opt.warp_ani else None, \
                 data['ani_image'].squeeze(1) if opt.warp_ani else None, \
                 None, None, None]
    synthesized_image, fake_raw_img, warped_img, flow, weight, _, _, ref_label, ref_image, img_ani = model(data_list, ref_idx_fix=ref_idx_fix)
    

    visual_list = []
    for i in range(opt.n_shot):
        visual_list += [('ref_img_{}'.format(i), util.tensor2im(data['ref_image'][:, i:i+1]))]
    visual_list += [('warping_ref_lmark', util.tensor2im(data['warping_ref_lmark'])),
                    ('warping_ref_img', util.tensor2im(data['warping_ref'])),
                    ('target_label', util.visualize_label(opt, data['tgt_label'])),
                    ('target_image', util.tensor2im(data['tgt_image'])),
                    ('synthesized_image', util.tensor2im(synthesized_image)),
                    ('ani_syn_image', util.tensor2im(img_ani)),
                    ('ref_warped_images', util.tensor2im(warped_img[0][-1], tile=True)),
                    ('ref_weights', util.tensor2im(weight[0][-1], normalize=False, tile=True)),
                    ('raw_image', util.tensor2im(fake_raw_img)),
                    ('ani_warped_images', util.tensor2im(warped_img[2][-1], tile=True) if warped_img[2] is not None else None),
                    ('ani_weights', util.tensor2im(weight[2][-1], normalize=False, tile=True) if weight[2] is not None else None),
                    ('ani_flow', util.tensor2flow(flow[2][-1], tile=True) if flow[2] is not None else None),
                    ('ref_flow', util.tensor2flow(flow[0][-1], tile=True)),
                    ('ani_image', util.tensor2im(data['ani_image'])),
                    ('ani_lmark', util.tensor2im(data['ani_lmark'])),
                    ('cropped_image', util.tensor2im(data['cropped_images'])),
                    ('cropped_lmark', util.tensor2im(data['cropped_lmarks'])),
                    ]
    visuals = OrderedDict(visual_list)

    print('process image... %s' % img_path)

    # for image save
    img_path_base = os.path.join(*(img_path[0].split('/')[:-3]))
    img_save_name = img_path[0].split('/')[-3:]
    img_save_name = "{}_{}_{}_{}".format(img_save_name[0], img_save_name[1], data['target_id'][0].tolist(), img_save_name[2])

    visualizer.save_images(webpage, visuals, [os.path.join(img_path_base, img_save_name)])

webpage.save()