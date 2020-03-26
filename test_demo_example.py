
import os
import numpy as np
import torch
import cv2
from collections import OrderedDict
from PIL import Image
import pickle as pkl
import math

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

def set_tgt_param(root, opt):
    if opt.dataset_name == 'vox':
        assert opt.tgt_video_path is not None
        opt.tgt_lmarks_path = opt.tgt_video_path[:-12]+"_aligned.npy"
        opt.tgt_rt_path = opt.tgt_video_path[:-12]+"_aligned_rt.npy"
        opt.tgt_ani_path = opt.tgt_video_path[:-12]+"_aligned_ani.mp4"
    elif opt.dataset_name == 'grid':
        assert opt.tgt_video_path is not None
        opt.tgt_lmarks_path = opt.tgt_video_path[:-9]+"_original.npy"
        opt.tgt_rt_path = opt.tgt_video_path[:-9]+"_rt.npy"
        opt.tgt_ani_path = None

def set_ref_param(root, opt):
    if opt.dataset_name == 'vox':
        assert opt.ref_video_path is not None
        opt.ref_front_path = opt.ref_video_path[:-12]+"_aligned.npy"
        opt.ref_lmarks_path = opt.ref_video_path[:-12]+"_aligned.npy"
        opt.ref_rt_path = opt.ref_video_path[:-12]+"_aligned_rt.npy"
        if opt.finetune:
            opt.ref_ani_path = opt.ref_video_path[:-12]+"_aligned_ani.mp4"
    elif opt.dataset_name == 'grid':
        assert opt.ref_video_path is not None
        opt.ref_front_path = None
        opt.ref_lmarks_path = opt.ref_video_path[:-9]+"_original.npy"
        opt.ref_rt_path = opt.tgt_video_path[:-9]+"_rt.npy"

def get_audio_path(opt):
    if opt.dataset_name == 'vox':
        audio_path = opt.tgt_video_path.replace('video', 'audio')[:-12]+'.wav'
    elif opt.dataset_name == 'grid':
        audio_path = opt.tgt_video_path.replace('video', 'audio')[:-9]+'.wav'
    return audio_path

opt = TestOptions().parse()

# preprocess
save_name = opt.name
if opt.dataset_name == 'lrs':
    save_name = 'lrs'
if opt.dataset_name == 'lrw':
    save_name = 'lrw'

save_root = os.path.join('test', save_name, 'finetune_front_{}'.format(opt.n_shot), '{}_shot_epoch_{}'.format(opt.n_shot, opt.which_epoch))

### setup models
model = create_model(opt)
model.eval()

# setup dataset
root = opt.dataroot
ref_root = opt.ref_dataroot

# set up parameters
set_tgt_param(root, opt)
set_ref_param(ref_root, opt)

### setup dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# test
ref_idx_fix = None
for i, data in enumerate(dataset):
    if i >= len(dataset): break
    img_path = data['path']
    if not opt.warp_ani:
        data.update({'ani_image':None, 'ani_lmark':None, 'cropped_images':None, 'cropped_lmarks':None })
    if "warping_ref" not in data:
        data.update({'warping_ref': data['ref_image'][:, :1], 'warping_ref_lmark': data['ref_label'][:, :1]})

    # finetune
    if i == 0 and opt.finetune:
        iteration = max(10, opt.n_shot * 10)
        model.finetune_call_multi(tgt_label_list=[data['ref_label']], tgt_image_list=[data['ref_image']],\
                                  ref_label_list=[data['ref_label']], ref_image_list=[data['ref_image']], \
                                  warp_ref_lmark_list=[data['warping_ref_lmark']], warp_ref_img_list=[data['warping_ref']], \
                                  ani_lmark_list=[data['ori_ani_lmark']], ani_img_list=[data['ori_ani_image']], \
                                  iterations=iteration)

    img_path = data['path']
    data_list = [data['tgt_label'], data['tgt_image'], None, None, None, None, \
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
        util.tensor2im(data['tgt_image']), \
        util.tensor2im(synthesized_image), \
        util.tensor2im(fake_raw_img), \
        util.tensor2im(warped_img[0]), \
        util.tensor2im(weight[0]), \
        util.tensor2im(warped_img[2]), \
        util.tensor2im(weight[2])
    ]
    compare_image = np.hstack([v for v in visuals if v is not None])

    # save directory
    if i == 0:
        img_path = data['path']
        img_id = "finetune_{}_{}_{}".format(img_path[0].split('/')[-3], img_path[0].split('/')[-2], img_path[0].split('/')[-1][:-4])
        img_dir = os.path.join(save_root,  img_id)
        if not os.path.exists(img_dir):
                os.makedirs(img_dir)

    img_id = "{}_{}_{}".format(img_path[0].split('/')[-3], img_path[0].split('/')[-2], img_path[0].split('/')[-1][:-4])
    img_dir = os.path.join(save_root,  img_id)
    img_name = "%05d.jpg"%data['index'][0]

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

# combine into video (save for compare)
v_n = os.path.join(img_dir, 'test.mp4')
image_to_video(sample_dir = img_dir, video_name = v_n)
audio_path = get_audio_path(opt)
add_audio(os.path.join(img_dir, 'test.mp4'), audio_path)