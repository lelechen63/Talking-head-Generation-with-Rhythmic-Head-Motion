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


opt = TestOptions().parse()

### setup models
model = create_model(opt)
model.eval()

# fake_root = "/home/cxu-serve/p1/common/tmp/atnet_raw_pca_test"
# fake_root = '/u/lchen63/Project/face_tracking_detection/eccv2020/sample/atnet_raw_pca_test'
fake_root = '/u/lchen63/Project/face_tracking_detection/eccv2020/sample/atnet_raw_pca_test_lrw'
files = [f for f in os.listdir(fake_root) if f[-3:]=='npy'][:opt.how_many]
# files = []
# select_fs = ['s14', 's15']
# for f in os.listdir(fake_root):
#     if f[-3:]=='npy' and f.split('__')[0] in select_fs:
#         files.append(f)
# files = files[:opt.how_many]

# files = ["s13__pbbo3a_front.npy"]
# files = files[:
# pdb.set_trace()

real_root = os.path.join(opt.dataroot, 'align')
# audio_root = os.path.join(opt.dataroot, 'audio')
audio_root = '/home/cxu-serve/p1/common/lrw/audio/'
save_root = os.path.join('evaluation_store', opt.name, 'fake_lmark')

visualizer = Visualizer(opt)
webpage = html.HTML(save_root, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch), infer=True)

paths = [None, None, None]
for file_id, file in enumerate(tqdm(files)):
    # if file_id >= opt.how_many: break

    print('process {} ...'.format(file))

    # paths[0] = file.split('__')[0]
    # paths[1] = file.split('__')[1].split('_')[0]
    paths = ['s23', 'lrav4p']
    word, name = file.split('_')[2], file.split('_')[3]

    # target
    opt.tgt_video_path = os.path.join(real_root, paths[0], paths[1]+"_crop.mp4")
    opt.tgt_lmarks_path = os.path.join(fake_root, file)
    # opt.tgt_lmarks_path = os.path.join(real_root, paths[0], paths[1]+"_original.npy")
    opt.tgt_rt_path = None
    opt.tgt_ani_path = None
    audio_tgt_path = os.path.join(audio_root, word, "test", "{}_{}.wav".format(word, name))
    # reference
    ref_paths = paths
    opt.ref_front_path = None
    opt.ref_video_path = opt.tgt_video_path
    opt.ref_lmarks_path = os.path.join(real_root, ref_paths[0], ref_paths[1]+"_original.npy")
    opt.ref_rt_path = None
    opt.ref_ani_id = None

    ### setup dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # test
    # ref_idx_fix = torch.zeros([opt.batchSize])
    ref_idx_fix = None
    for i, data in enumerate(dataset):
        if i >= len(dataset): break
        img_path = data['path']
        if not opt.warp_ani:
            data.update({'ani_image':None, 'ani_lmark':None, 'cropped_images':None, 'cropped_lmarks':None })
        if "warping_ref" not in data:
            data.update({'warping_ref': data['ref_image'][:, 0], 'warping_ref_lmark': data['ref_label'][:, 0]})

        img_path = data['path']
        data_list = [data['tgt_label'], data['tgt_image'], None, None, None, None, \
                    data['ref_label'], data['ref_image'], \
                    data['warping_ref_lmark'], \
                    data['warping_ref'], \
                    data['ani_lmark'].squeeze(1) if opt.warp_ani else None, \
                    data['ani_image'].squeeze(1) if opt.warp_ani else None, \
                    None, None, None]
        synthesized_image, fake_raw_img, warped_img, flow, weight, _, _, _, _, _ = model(data_list, ref_idx_fix=ref_idx_fix)
        
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

        synthesized_image = util.tensor2im(synthesized_image)
        tgt_image = util.tensor2im(data['tgt_image'])


        # img_id = "{}_{}_{}".format(img_path[0].split('/')[-3], img_path[0].split('/')[-2], img_path[0].split('/')[-1][:-4])
        img_id = file[:-4]
        img_dir = os.path.join(save_root,  img_id)
        img_name = "%05d.jpg"%data['index'][0]
        img_test_dir = os.path.join(save_root, 'test')

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        image_pil = Image.fromarray(compare_image).convert('RGB')
        image_pil.save(os.path.join(img_dir, img_name))

        if not os.path.exists(img_test_dir):
            os.makedirs(img_test_dir)
        image_pil = Image.fromarray(synthesized_image).convert('RGB')
        image_pil.save(os.path.join(img_test_dir, img_name))

        # save reference
        if i == 0:
            if not os.path.exists(os.path.join(img_dir, 'reference')):
                os.makedirs(os.path.join(img_dir, 'reference'))
            for ref_img_id in range(data['ref_image'].shape[1]):
                ref_img = util.tensor2im(data['ref_image'][0, ref_img_id])
                ref_img = Image.fromarray(ref_img).convert('RGB')
                ref_img.save(os.path.join(img_dir, 'reference', 'ref_{}.jpg').format(ref_img_id))
            warp_ref_img = util.tensor2im(data['warping_ref'][0])
            warp_ref_img = Image.fromarray(warp_ref_img).convert('RGB')
            warp_ref_img.save(os.path.join(img_dir, 'reference', 'warping_ref.jpg'))

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

        # visual_list = []
        # for i in range(opt.n_shot):
        #     visual_list += [('warping_ref_lmark', util.tensor2im(data['warping_ref_lmark'])),
        #                     ('warping_ref_img', util.tensor2im(data['warping_ref'])),
        #                     ('target_label', util.visualize_label(opt, data['tgt_label'])),
        #                     ('target_image', util.tensor2im(data['tgt_image'])),
        #                     ('synthesized_image', util.tensor2im(synthesized_image)),
        #                     ('ref_warped_images', util.tensor2im(warped_img[0][-1], tile=True)),
        #                     ('ref_weights', util.tensor2im(weight[0][-1], normalize=False, tile=True)),
        #                     ('raw_image', util.tensor2im(fake_raw_img)),
        #                     ('ani_warped_images', util.tensor2im(warped_img[2][-1], tile=True) if warped_img[2] is not None else None),
        #                     ('ani_weights', util.tensor2im(weight[2][-1], normalize=False, tile=True) if weight[2] is not None else None),
        #                     ('ani_flow', util.tensor2flow(flow[2][-1], tile=True) if flow[2] is not None else None),
        #                     ('ref_flow', util.tensor2flow(flow[0][-1], tile=True)),
        #                     ]
        # visuals = OrderedDict(visual_list)

        # # for image save
        # visualizer.save_images(webpage, visuals, [os.path.join(save_root, file)])

    # combine into video
    # mmcv.frames2video(img_dir, os.path.join(img_dir, 'test.mp4'))
    image_to_video(img_dir, os.path.join(img_dir, 'test.mp4'))
    add_audio(os.path.join(img_dir, 'test.mp4'), audio_tgt_path)
    # mmcv.frames2video(img_test_dir, os.path.join(img_test_dir, '{}.mp4'.format(file.split('.')[0])))
    image_to_video(img_test_dir, os.path.join(img_test_dir, '{}.mp4'.format(file.split('.')[0])))
    # add_audio(os.path.join(img_test_dir, '{}.mp4'.format(file.split('.')[0])), audio_tgt_path)
    for f in os.listdir(img_test_dir):
        if f.split('.')[1] != "mp4":
            os.remove(os.path.join(img_test_dir, f))