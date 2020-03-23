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

def add_flip(image, landmark):
    flip_image = image.flip(4)
    flip_landmark = landmark.flip(4)
    return flip_image, flip_landmark

def get_param(root, opt):
    if opt.dataset_name == 'lisa':
        # target
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__lisa2__rotated.npy")
        opt.tgt_lmarks_path = os.path.join(root, "demo_00025_aligned__rotated.npy")
        # opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        opt.tgt_rt_path = os.path.join(root, "00025_aligned_rt.npy")
        # opt.tgt_ani_path = os.path.join(root, "lisa2_ani2.mp4")
        opt.tgt_ani_path = os.path.join(root, "00025_aligned__lisa2_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'lisa2_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'lisa2_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'lisa2_original.npy')
        opt.ref_rt_path = os.path.join(root, 'lisa2_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/lisa2_original.png')
        opt.ref_ani_id = 0

    elif opt.dataset_name == 'vincent':
        # target
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        # opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        # opt.tgt_ani_path = os.path.join(root, "vincent2_ani2.mp4")
        opt.tgt_lmarks_path = os.path.join(root, "demo_00025_aligned__rotated.npy")
        opt.tgt_rt_path = os.path.join(root, "00025_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, "00025_aligned__vincent2_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'vincent2_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'vincent2_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'vincent2_original.npy')
        opt.ref_rt_path = os.path.join(root, 'vincent2_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/vincent2_original.png')
        opt.ref_ani_id = 0

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
        opt.ref_ani_path = os.path.join('/u/lchen63/Project/face_tracking_detection/eccv2020/face-tools/tempp_00005/mulan2_original.png')
        opt.ref_ani_id = 0

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

    elif opt.dataset_name == 'groot':
        # target
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        opt.tgt_lmarks_path = os.path.join(root, "2_1__groot1__rotated.npy")
        opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        opt.tgt_ani_path = os.path.join(root, "groot1_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'groot1_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'groot1_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'groot1_original.npy')
        opt.ref_rt_path = os.path.join(root, 'groot1_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/groot1_original.png')
        opt.ref_ani_id = 0

    elif opt.dataset_name == 'hulk':
        # target
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__hulk1__rotated.npy")
        # opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        # opt.tgt_ani_path = os.path.join(root, "hulk1_ani2.mp4")
        opt.tgt_lmarks_path = os.path.join(root, "demo_00025_aligned__rotated.npy")
        opt.tgt_rt_path = os.path.join(root, "00025_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, "00025_aligned__hulk1_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'hulk1_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'hulk1_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'hulk1_original.npy')
        opt.ref_rt_path = os.path.join(root, 'hulk1_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/hulk1_original.png')
        opt.ref_ani_id = 0

    elif opt.dataset_name == 'superman':
        # target
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__superman1__rotated.npy")
        # opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        # opt.tgt_ani_path = os.path.join(root, "superman1_ani2.mp4")
        opt.tgt_lmarks_path = os.path.join(root, "demo_00025_aligned__rotated.npy")
        opt.tgt_rt_path = os.path.join(root, "00025_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, "00025_aligned__superman1_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'superman1_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'superman1_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'superman1_original.npy')
        opt.ref_rt_path = os.path.join(root, 'superman1_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/superman1_original.png')
        opt.ref_ani_id = 0

    elif opt.dataset_name == 'picasso':
        # target
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        opt.tgt_lmarks_path = os.path.join(root, "2_1__picasso1__rotated.npy")
        opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        opt.tgt_ani_path = os.path.join(root, "picasso1_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'picasso1_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'picasso1_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'picasso1_original.npy')
        opt.ref_rt_path = os.path.join(root, 'picasso1_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/picasso1_original.png')
        opt.ref_ani_id = 0

    elif opt.dataset_name == 'turing':
        # target
        opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__turing1__rotated.npy")
        opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        opt.tgt_ani_path = os.path.join(root, "turing1_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'turing1_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'turing1_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'turing1_original.npy')
        opt.ref_rt_path = os.path.join(root, 'turing1_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/turing1_original.png')
        opt.ref_ani_id = 0

        audio_tgt_path = os.path.join(root, "2_1.wav")

    elif opt.dataset_name == 'frid':
        # target
        opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__frid1__rotated.npy")
        opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        opt.tgt_ani_path = os.path.join(root, "frid1_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'frid1_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'frid1_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'frid1_original.npy')
        opt.ref_rt_path = os.path.join(root, 'frid1_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/frid1_original.png')
        opt.ref_ani_id = 0

    elif opt.dataset_name == 'self1':
        # target
        opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__frid1__rotated.npy")
        opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        opt.tgt_ani_path = os.path.join(root, "self1_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'self1_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'self1_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'self1_original.npy')
        opt.ref_rt_path = os.path.join(root, 'self1_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/self1_original.png')
        opt.ref_ani_id = 0

    elif opt.dataset_name == 'self2':
        # target
        opt.tgt_lmarks_path = os.path.join(root, "2_1__original2.npy")
        # opt.tgt_lmarks_path = os.path.join(root, "2_1__frid1__rotated.npy")
        opt.tgt_rt_path = os.path.join(root, "2_1__rt2.npy")
        opt.tgt_ani_path = os.path.join(root, "self2_ani2.mp4")
        # reference
        opt.ref_front_path = os.path.join(root, 'self2_original_front.npy')
        opt.ref_video_path = os.path.join(root, 'self2_crop.png')
        opt.ref_lmarks_path = os.path.join(root, 'self2_original.npy')
        opt.ref_rt_path = os.path.join(root, 'self2_original_rt.npy')
        opt.ref_ani_path = os.path.join('/home/cxu-serve/p1/common/demo_ani/self2_original.png')
        opt.ref_ani_id = 0

    audio_tgt_path = os.path.join(root, "2_1_cut.wav")

    return audio_tgt_path

opt = TestOptions().parse()

opt.n_shot = 2
### setup models
model = create_model(opt)
model.eval()
opt.n_shot = 1

root = opt.dataroot

save_name = opt.name
save_root = os.path.join('interest_img', save_name, '{}_shot_test_{}'.format(opt.n_shot, opt.dataset_name))


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

    # try
    opt.n_shot = 2
    if opt.finetune:
        opt.finetune_shot = 2
        # augment target image
        flip_image, flip_landmark = add_flip(data['ref_image'], data['ref_label'])
        tgt_labels = torch.cat([data['ref_label'], flip_landmark], axis=1)
        tgt_images =  torch.cat([data['ref_image'], flip_image], axis=1)
        # augment animation image
        flip_image, flip_landmark = add_flip(data['ori_ani_image'], data['ori_ani_lmark'])
        ani_lmark = torch.cat([data['ori_ani_lmark'], flip_landmark], axis=1)
        ani_image = torch.cat([data['ori_ani_image'], flip_image], axis=1)
        # augment warping reference image
        flip_image, flip_landmark = add_flip(data['warping_ref'], data['warping_ref_lmark'])
        warp_ref_lmark = torch.cat([data['warping_ref_lmark'], flip_landmark], axis=1)
        warp_ref =  torch.cat([data['warping_ref'], flip_image], axis=1)

        # opt.finetune_shot = 7
        # tgt_labels = data['ref_label']
        # tgt_images = data['ref_image']
        # warp_ref_lmark = data['warping_ref_lmark']
        # warp_ref = data['warping_ref']
        # ani_lmark = data['ori_ani_lmark']
        # ani_image = data['ori_ani_image']
        model.finetune_call(tgt_labels=tgt_labels, tgt_images=tgt_images, \
                            ref_labels=tgt_labels, ref_images=tgt_images, \
                            warp_ref_lmark=warp_ref_lmark, warp_ref_img=warp_ref, \
                            ani_lmark=ani_lmark, ani_img=ani_image)

    # augment reference image
    flip_image, flip_landmark = add_flip(data['ref_image'], data['ref_label'])
    ref_labels = torch.cat([data['ref_label'], flip_landmark], axis=1)
    ref_images =  torch.cat([data['ref_image'], flip_image], axis=1)
    # ref_labels = data['ref_label']
    # ref_images = data['ref_image']

    data_list = [data['tgt_label'], None, None, None, None, None, \
                ref_labels, ref_images, \
                data['warping_ref_lmark'].squeeze(1) if data['warping_ref_lmark'] is not None else None, \
                data['warping_ref'].squeeze(1) if data['warping_ref'] is not None else None, \
                data['ani_lmark'].squeeze(1) if opt.warp_ani else None, \
                data['ani_image'].squeeze(1) if opt.warp_ani else None, \
                None, None, None]
    synthesized_image, fake_raw_img, warped_img, _, weight, _, _, _, _, _ = model(data_list, ref_idx_fix=ref_idx_fix)
    
    # try
    opt.n_shot = 1

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
    # if i == 0:
    #     if not os.path.exists(os.path.join(img_dir, 'reference')):
    #         os.makedirs(os.path.join(img_dir, 'reference'))
    #     for ref_img_id in range(data['ref_image'].shape[1]):
    #         ref_img = util.tensor2im(data['ref_image'][0, ref_img_id])
    #         ref_img = Image.fromarray(ref_img)
    #         ref_img.save(os.path.join(img_dir, 'reference', 'ref_{}.jpg').format(ref_img_id))

    #     ori_ani = util.tensor2im(data['ori_ani_image'][0])
    #     ori_ani_lmark = util.tensor2im(data['ori_ani_lmark'][0])
    #     save_img = np.hstack([ori_ani, ori_ani_lmark])
    #     save_img = Image.fromarray(save_img)
    #     save_img.save(os.path.join(img_dir, 'reference', 'ori_ani.jpg'))
    if i == 0:
        if not os.path.exists(os.path.join(img_dir, 'reference')):
            os.makedirs(os.path.join(img_dir, 'reference'))
        for ref_img_id in range(ref_images.shape[1]):
            ref_img = util.tensor2im(ref_images[0, ref_img_id])
            ref_img = Image.fromarray(ref_img)
            ref_img.save(os.path.join(img_dir, 'reference', 'ref_{}.jpg').format(ref_img_id))

        if 'ori_ani_image' in data:
            ori_ani = util.tensor2im(data['ori_ani_image'][0])
            ori_ani_lmark = util.tensor2im(data['ori_ani_lmark'][0])
            save_img = np.hstack([ori_ani, ori_ani_lmark])
            save_img = Image.fromarray(save_img)
            save_img.save(os.path.join(img_dir, 'reference', 'ori_ani.jpg'))

    # save for evaluation
    # if opt.evaluate:
    #     if not os.path.exists(os.path.join(img_dir, 'real')):
    #         os.makedirs(os.path.join(img_dir, 'real'))
    #     img_path = os.path.join(img_dir, 'real', '{}_{}_image.jpg'.format(data['target_id'][0], 'real'))
    #     image_pil = Image.fromarray(tgt_image)
    #     image_pil.save(img_path)

    #     if not os.path.exists(os.path.join(img_dir, 'synthesized')):
    #         os.makedirs(os.path.join(img_dir, 'synthesized'))
    #     img_path = os.path.join(img_dir, 'synthesized', '{}_{}_image.jpg'.format(data['target_id'][0], 'synthesized'))
    #     image_pil = Image.fromarray(synthesized_image)
    #     image_pil.save(img_path)

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