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
    img_path = data['path']   
    data_list = [data['tgt_label'], data['tgt_image'], None, None, data['ref_label'], data['ref_image'], None, None]
    synthesized_image, fake_raw_img, warped_img, flow, weight, _, ref_idx = model(data_list, ref_idx_fix=ref_idx_fix)
    
    # synthesized_image = util.tensor2im(synthesized_image)    
    # tgt_image = util.tensor2im(data['tgt_image'])    
    # ref_image = util.tensor2im(data['ref_image'], tile=True)
    # pdb.set_trace()
    visuals = OrderedDict([ ('target_label', util.tensor2im(data['tgt_label'])),
                            ('synthesized_image', util.tensor2im(synthesized_image)),
                            ('target_image', util.tensor2im(data['tgt_image'])),
                            ('ref_image', util.tensor2im(data['ref_image'])),
                            ('raw_image', util.tensor2im(fake_raw_img)),
                            ('warped_image', util.tensor2im(warped_img[0])),
                            ('flow', util.tensor2flow(flow[0])),
                            ('weight', util.tensor2im(weight[0], normalize=False))])

    print('process image... %s' % img_path)

    # for image save
    img_path_base = os.path.join(img_path.split('/')[:-3])
    img_save_name = img_save_name.split('/')[-3:]
    img_save_name = "{}_{}_{}".format(img_save_name[0], img_save_name[1], img_save_name[2])

    visualizer.save_images(webpage, visuals, os.path.join(img_path_base, img_save_name))

webpage.save()