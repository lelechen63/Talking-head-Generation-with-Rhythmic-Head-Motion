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

def add_audio(video_name, audio_dir):
    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
    #ffmpeg -i /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/new/audio/obama.wav -codec copy -c:v libx264 -c:a aac -b:a 192k  -shortest -y /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mov
    # ffmpeg -i gan_r_high_fake.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/audio/obama.wav -vcodec copy  -acodec copy -y   gan_r_high_fake.mov

    print (command)
    os.system(command)

def image_to_video(sample_dir = None, video_name = None):
    
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.jpg -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 
    #ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
    print (command)
    os.system(command)

def get_param(root, opt):
    if opt.dataset_name == 'lisa':
        # target
        opt.tgt_lmarks_path = os.path.join(root, "00216_aligned.npy")
        opt.tgt_rt_path = os.path.join(root, "00216_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, "lisa2_ani.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'lisa2_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'lisa2_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'lisa2_original.npy')
        opt.ref_rt_path = os.path.join(root, 'lisa2_original_rt.npy')
        opt.ref_ani_id = 0

        audio_tgt_path = os.path.join(root, "f_f.wav")

    elif opt.dataset_name == 'vincent':
        # target
        opt.tgt_lmarks_path = os.path.join(root, "00216_aligned.npy")
        opt.tgt_rt_path = os.path.join(root, "00216_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, "vincent2_ani.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'vincent2_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'vincent2_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'vincent2_original.npy')
        opt.ref_rt_path = os.path.join(root, 'vincent2_original_rt.npy')
        opt.ref_ani_id = 0

        audio_tgt_path = os.path.join(root, "f_f.wav")

    elif opt.dataset_name == 'mulan':
        # target
        opt.tgt_lmarks_path = os.path.join(root, "00216_aligned.npy")
        opt.tgt_rt_path = os.path.join(root, "00216_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, "mulan2_ani.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'mulan2_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'mulan2_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'mulan2_original.npy')
        opt.ref_rt_path = os.path.join(root, 'mulan2_original_rt.npy')
        opt.ref_ani_id = 0

        audio_tgt_path = os.path.join(root, "f_f.wav")

    elif opt.dataset_name == 'david':
        # target
        opt.tgt_lmarks_path = os.path.join(root, "00216_aligned.npy")
        opt.tgt_rt_path = os.path.join(root, "00216_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, "david_ani.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'david_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'david_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'david_original.npy')
        opt.ref_rt_path = os.path.join(root, 'david_original_rt.npy')
        opt.ref_ani_id = 0

        audio_tgt_path = os.path.join(root, "f_f.wav")

    return audio_tgt_path

opt = TestOptions().parse()

### setup models
model = create_model(opt)
model.eval()

root = opt.dataroot

save_name = opt.name
save_root = os.path.join('evaluation_store', save_name, '{}_shot_test_{}'.format(opt.n_shot, opt.dataset_name))


audio_tgt_path = get_param(root, opt)

### setup dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# test
# ref_idx_fix = torch.zeros([opt.batchSize])
print('generating image')
ref_idx_fix = None
for i, data in enumerate(tqdm(dataset)):

    # pdb.set_trace()

    # if i >= 10: break
    if i >= len(dataset): break
    if not opt.warp_ani:
        data.update({'ani_image':None, 'ani_lmark':None, 'cropped_images':None, 'cropped_lmarks':None })
    if "warping_ref" not in data:
        data.update({'warping_ref': data['ref_image'][:, :1], 'warping_ref_lmark': data['ref_label'][:, :1]})

    data_list = [data['tgt_label'], None, None, None, None, None, \
                data['ref_label'], data['ref_image'], \
                data['warping_ref_lmark'].squeeze(1) if data['warping_ref_lmark'] is not None else None, \
                data['warping_ref'].squeeze(1) if data['warping_ref'] is not None else None, \
                data['ani_lmark'].squeeze(1) if opt.warp_ani else None, \
                data['ani_image'].squeeze(1) if opt.warp_ani else None, \
                None, None, None]
    synthesized_image, fake_raw_img, warped_img, _, weight, _, _, _, _, _ = model(data_list, ref_idx_fix=ref_idx_fix)
    
    # save compare
    visuals = [
        util.tensor2im(data['tgt_label']), \
        util.tensor2im(synthesized_image), \
        util.tensor2im(fake_raw_img), \
        util.tensor2im(warped_img[0]), \
        util.tensor2im(weight[0]), \
        util.tensor2im(warped_img[2]), \
        util.tensor2im(weight[2]), \
        util.tensor2im(data['ani_image'])
    ]
    compare_image = np.hstack([v for v in visuals if v is not None])

    img_dir = os.path.join(save_root,  opt.dataset_name)
    img_name = "%05d.jpg"%data['index'][0]

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    image_pil = Image.fromarray(compare_image)
    image_pil.save(os.path.join(img_dir, img_name))

    # save for test
    test_syn_image = util.tensor2im(synthesized_image)
    img_test_dir = os.path.join(save_root, 'test')
    if not os.path.exists(img_test_dir):
        os.makedirs(img_test_dir)
    image_pil = Image.fromarray(test_syn_image)
    image_pil.save(os.path.join(img_test_dir, img_name))

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
        img_path = os.path.join(img_dir, 'real', '{}_{}_image.jpg'.format(data['target_id'][0], 'real'))
        image_pil = Image.fromarray(tgt_image)
        image_pil.save(img_path)

        if not os.path.exists(os.path.join(img_dir, 'synthesized')):
            os.makedirs(os.path.join(img_dir, 'synthesized'))
        img_path = os.path.join(img_dir, 'synthesized', '{}_{}_image.jpg'.format(data['target_id'][0], 'synthesized'))
        image_pil = Image.fromarray(synthesized_image)
        image_pil.save(img_path)

    # print('process image... %s' % img_path)

# combine into video (save for compare)
v_n = os.path.join(img_dir, 'test.mp4')
image_to_video(sample_dir = img_dir, video_name = v_n)
add_audio(os.path.join(img_dir, 'test.mp4'), audio_tgt_path)
# combine into video (save for test)
v_n = os.path.join(img_test_dir, '{}.mp4'.format('lisa'))
image_to_video(sample_dir = img_test_dir, video_name = v_n)
for f in os.listdir(img_test_dir):
    if f.split('.')[1] != "mp4":
        os.remove(os.path.join(img_test_dir, f))