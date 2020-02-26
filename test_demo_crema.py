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

def sel_for_crema(pickle_data, sel_emotion):
    sel_pickle = []
    human_id = {}
    for paths in pickle_data:
        h_id, word, emo, emo_lvl = paths[0][:-11].split('_')
        # select emotion
        if emo in sel_emotion:
            # select person id
            if h_id not in human_id:
                sel_pickle.append(paths)
                human_id[h_id] = ['{}_{}'.format(emo, emo_lvl)]
            # select emotion and level pair
            elif '{}_{}'.format(emo, emo_lvl) not in human_id[h_id]:
                sel_pickle.append(paths)
                human_id[h_id].append('{}_{}'.format(emo, emo_lvl))
    
    return sel_pickle

def add_audio(video_name, audio_dir):
    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
    #ffmpeg -i /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/new/audio/obama.wav -codec copy -c:v libx264 -c:a aac -b:a 192k  -shortest -y /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mov
    # ffmpeg -i gan_r_high_fake.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/audio/obama.wav -vcodec copy  -acodec copy -y   gan_r_high_fake.mov

    print (command)
    os.system(command)

def image_to_video(sample_dir = None, video_name = None):
    
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.png -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 
    #ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
    print (command)
    os.system(command)

def get_param(root, pickle_data, pick_id, opt):
    paths = pickle_data[pick_id]
    if opt.dataset_name == 'crema':
        # target
        opt.tgt_video_path = os.path.join(root, 'VideoFlash', paths[0][:-10]+"_crop.mp4")
        opt.tgt_lmarks_path = os.path.join(root, 'VideoFlash', paths[0][:-10]+"_original.npy")
        opt.tgt_rt_path = os.path.join(root, 'VideoFlash', paths[0][:-10]+"_rt.npy")
        opt.tgt_ani_path = None
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, 'VideoFlash', paths[0][:-10]+"_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = None

        audio_tgt_path = os.path.join(root, 'AudioWAV', paths[0][:-11]+".wav")

    return audio_tgt_path

opt = TestOptions().parse()

### setup models
model = create_model(opt)
model.eval()

root = opt.dataroot
if opt.dataset_name == 'crema':
    _file = open(os.path.join(root, 'pickle','train_lmark2img.pkl'), "rb")
pickle_data = pkl.load(_file)
_file.close()

# select certain human and emotion
sel_emo = ['ANG']
start_id = int(0.8 * len(pickle_data))
pickle_data = pickle_data[start_id:]
pickle_data = sel_for_crema(pickle_data, sel_emo)

save_name = opt.name
save_root = os.path.join('evaluation_store', save_name, 'for_paper', '{}_shot_test'.format(opt.n_shot))
# pick_ids = np.random.choice(list(range(len(pickle_data))), size=opt.how_many)
end = int(len(pickle_data))
# pick_ids = range(0, end, end//opt.how_many)
# pick_ids = range(0, opt.how_many)
pick_ids = range(0, len(pickle_data))
# pick_files = ['s14', 's15']
# pick_files = ['1075_IWL_FEA_XX_', '1090_IOM_FEA_XX_', '1088_ITH_HAP_XX_', '1091_IWL_NEU_XX_', \
#               '1085_TIE_HAP_XX_', '1075_TIE_HAP_XX_', '1077_WSI_FEA_XX__', '1089_IWL_ANG_XX_']
# pick_files = np.asarray([['id00081', '2xYrsnvtUWc', '00002'], ['id00081', '2xYrsnvtUWc', '00004'], ['id01000', '0lmrq0quo9M', '00001']])

# pick_files = np.asarray([['id04094', '2sjuXzB2I1M', '00025'], \
#                ['id00081', '2xYrsnvtUWc', '00002'], \
#                ['id00081', '2xYrsnvtUWc', '00004'], \
#                ['id01000', '0lmrq0quo9M', '00001']])
# pick_ids = range(0, len(pickle_data))

for pick_id in tqdm(pick_ids):
    print('process {} ...'.format(pick_id))
    audio_tgt_path = get_param(root, pickle_data, pick_id, opt)

    paths = pickle_data[pick_id]

    ### setup dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # test
    # ref_idx_fix = torch.zeros([opt.batchSize])
    ref_idx_fix = None
    for i, data in enumerate(dataset):
        # if i >= 10: break
        if i >= len(dataset): break
        img_path = data['path']
        if not opt.warp_ani:
            data.update({'ani_image':None, 'ani_lmark':None, 'cropped_images':None, 'cropped_lmarks':None })
        if "warping_ref" not in data:
            data.update({'warping_ref': data['ref_image'][:, :1], 'warping_ref_lmark': data['ref_label'][:, :1]})
        # data.update({'warping_ref': data['ref_image'][:, :1], 'warping_ref_lmark': data['ref_label'][:, :1]})

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

        img_id = "{}_{}_{}".format(img_path[0].split('/')[-3], img_path[0].split('/')[-2], img_path[0].split('/')[-1][:-4])
        img_dir = os.path.join(save_root,  img_id)
        img_name = "%05d.png"%data['index'][0]

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
