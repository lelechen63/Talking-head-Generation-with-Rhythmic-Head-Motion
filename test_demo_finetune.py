
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

def get_param(root, pickle_data, pick_id, opt):
    paths = pickle_data[pick_id]

    # check shot
    ref_nums = opt.ref_img_id.split(',')
    if opt.n_shot % len(ref_nums) != 0:
        print('reference number error')
        exit(0)
    else:
        ref_nums = ref_nums * (opt.n_shot // len(ref_nums))
        opt.ref_img_id = ','.join(ref_nums)

    if opt.dataset_name == 'vox':
        # target
        opt.tgt_video_path = paths[0]+"_aligned.mp4"
        if opt.no_head_motion:
            opt.tgt_lmarks_path = paths[0]+"_aligned_front.npy"
        else:
            opt.tgt_lmarks_path = paths[0]+"_aligned.npy"
        opt.tgt_rt_path = paths[0]+"_aligned_rt.npy"
        opt.tgt_ani_path = paths[0]+"_aligned_ani.mp4"

        # reference
        ref_paths = paths
        opt.ref_ani_id = int(paths[-1])
        opt.ref_front_path = ref_paths[0]+"_aligned_front.npy"
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = paths[0]+"_aligned.npy"
        opt.ref_rt_path = opt.tgt_rt_path
        if opt.no_head_motion:
            opt.ref_img_id = str(opt.ref_ani_id)
            opt.n_shot = 1

        audio_tgt_path = paths[0]+".wav"

    return audio_tgt_path


def sel_data(pickle_data, opt):
    # option 1
    finetune_data = [data for data in pickle_data]
    pickle_data = [data for data in pickle_data]

    finetune_ids = [0]
    pickle_ids = [0]

    return finetune_data, finetune_ids, pickle_data, pickle_ids


opt = TestOptions().parse()

# preprocess
save_name = opt.name
save_root = os.path.join('extra_degree_result_ani', save_name, 'finetune_front_{}'.format(opt.finetune_shot), '{}_shot_epoch_{}'.format(opt.n_shot, opt.which_epoch))

### setup models
model = create_model(opt)
model.eval()

# setup dataset
root = opt.dataroot
pickle_files = np.load('vox_demo_ani.npy')
pickle_data = []

for paths in pickle_files:
    file = paths[0]
    audio_file = file.replace('video', 'audio').replace('_aligned.mp4', '.wav')
    landmark_file = file.replace('_aligned.mp4', '_aligned.npy')
    rt_file = file.replace('_aligned.mp4', '_aligned_rt.npy')
    if not os.path.exists(audio_file) or not os.path.exists(landmark_file) or not os.path.exists(rt_file):
        continue
    cur_data = [file.replace('_aligned.mp4', ''), paths[1], paths[2], paths[-1]]
    pickle_data.append(cur_data)

# for finetune
finetune_data, finetune_ids, pickle_data, pick_ids = sel_data(pickle_data, opt)

test_ref_id = opt.ref_img_id
test_shot = opt.n_shot
# finetune 
# opt.ref_img_id = ""
# opt.n_shot = opt.finetune_shot
# step = 2048 // opt.finetune_shot
# for i in range(0, opt.finetune_shot):
#     opt.ref_img_id += '{},'.format(i * step)
# opt.ref_img_id = opt.ref_img_id[:-1]

refs, warps, anis = [], [], []
for f_id in finetune_ids: 
    get_param(root, finetune_data, f_id, opt)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    data = next(iter(dataset))
    ref_labels, ref_images = data['ref_label'], data['ref_image']
    warping_ref_lmark, warping_ref = data['warping_ref_lmark'], data['warping_ref']
    if 'ori_ani_image' in data:
        anis.append([data['ori_ani_lmark'], data['ori_ani_image']])
    refs.append([ref_labels, ref_images])
    warps.append([warping_ref_lmark, warping_ref])

ref_label_list, ref_image_list = [], []
tgt_label_list, tgt_image_list = [], []
warp_label_list, warp_image_list = [], []
ani_label_list, ani_image_list = [], []
total_iterations = max(15, opt.finetune_shot*2)            # finetune
opt.n_shot = test_shot
for ref, warp in zip(refs, warps):
    ref_labels, ref_images = ref
    warping_ref_lmark, warping_ref = warp
    tgt_label_list.append(ref_labels)
    tgt_image_list.append(ref_images)
    ref_label_list.append(ref_labels[:,:opt.n_shot])
    ref_image_list.append(ref_images[:,:opt.n_shot])
    warp_label_list.append(warping_ref_lmark)
    warp_image_list.append(warping_ref)
if len(anis) != 0:
    for ani in anis:
        ani_label_list.append(ani[0])
        ani_image_list.append(ani[1])
else:
    ani_label_list, ani_image_list = None, None

model.finetune_call_multi(tgt_label_list, tgt_image_list, \
                ref_label_list, ref_image_list, \
                warp_label_list, warp_image_list, \
                ani_label_list, ani_image_list, \
                iterations=total_iterations)

img_path = data['path']
img_dir = os.path.join(save_root, 'finetune')

if not os.path.exists(img_dir):
    os.makedirs(img_dir)
for ref_id in range(len(refs)):
    ref_labels, ref_images = refs[ref_id]
    for ref_img_id in range(ref_images.shape[1]):
        ref_img = util.tensor2im(ref_images[0, ref_img_id])
        ref_label = util.tensor2im(ref_labels[0, ref_img_id])
        save_list = [ref_img, ref_label]
        if ani_label_list is not None:
            ani = anis[ref_id]
            ani_img = util.tensor2im(ani[0][0, ref_img_id]) 
            ani_label = util.tensor2im(ani[1][0, ref_img_id])
            save_list.append(ani_img)
            save_list.append(ani_label)
        save_img = np.hstack(save_list)
        save_img = Image.fromarray(save_img)
        save_img.save(os.path.join(img_dir, '{}_ref_{}.png').format(ref_id, ref_img_id))

# test
opt.ref_img_id = test_ref_id
opt.n_shot = test_shot
count = 0
for pick_id in tqdm(pick_ids):
    paths = pickle_data[pick_id]
    print('process {} ...'.format(pick_id))
    audio_tgt_path = get_param(root, pickle_data, pick_id, opt)

    ### setup dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # test
    ref_idx_fix = None
    for i, data in enumerate(dataset):
        if i >= 500: break
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
                    None,
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

        img_id = img_path[0].split('/')[-1][:-4]
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
                ref_img.save(os.path.join(img_dir, 'reference', 'ref_{}.png').format(ref_img_id))

    # combine into video (save for compare)
    v_n = os.path.join(img_dir, 'test.mp4')
    image_to_video(sample_dir = img_dir, video_name = v_n)
    add_audio(os.path.join(img_dir, 'test.mp4'), audio_tgt_path)