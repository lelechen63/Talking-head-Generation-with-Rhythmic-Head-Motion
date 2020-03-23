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


def get_good_file():
    pickle_data = []
    root_path = os.path.join('evaluation_store', 'good')
    # root_path = "/home/cxu-serve/p1/common/other/lrs_good"
    files = os.listdir(root_path)
    for f in files:
        pickle_data.append(f)

    # pickle_data = pickle_data[len(pickle_data)//2:]
    return pickle_data


def add_audio(video_name, audio_dir):
    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
    #ffmpeg -i /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/new/audio/obama.wav -codec copy -c:v libx264 -c:a aac -b:a 192k  -shortest -y /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mov
    # ffmpeg -i gan_r_high_fake.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/audio/obama.wav -vcodec copy  -acodec copy -y   gan_r_high_fake.mov

    print (command)
    os.system(command)

def image_to_video(sample_dir = None, video_name = None):
    
    command = 'ffmpeg -framerate 30  -i ' + sample_dir +  '/%05d.jpg -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 
    #ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
    print (command)
    os.system(command)

def get_param(root, pickle_data, pick_id, opt):
    paths = pickle_data[pick_id]
    if opt.dataset_name == 'vox':
        # target
        audio_package = 'unzip/test_video'
        opt.tgt_video_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned.mp4")
        if opt.no_head_motion:
            opt.tgt_lmarks_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned_front.npy")
        else:
            # opt.tgt_lmarks_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned.npy")
            # opt.tgt_lmarks_path = os.path.join("/home/cxu-serve/p1/common/other/vox_test", '{}__{}__{}_aligned_front_diff_rotated.npy'.format(paths[0], paths[1], paths[2]))
            opt.tgt_lmarks_path = os.path.join("/home/cxu-serve/p1/common/other/vox_test", '{}__{}__{}_aligned_front_diff.npy'.format(paths[0], paths[1], paths[2]))
        opt.tgt_rt_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned_ani.mp4")
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, audio_package, ref_paths[0], ref_paths[1], ref_paths[2]+"_aligned_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned.npy")
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = int(ref_paths[3])
        if opt.no_head_motion:
            opt.ref_img_id = str(opt.ref_ani_id)
            opt.n_shot = 1

        audio_tgt_path = os.path.join(root, 'unzip/test_audio', paths[0], paths[1], paths[2]+".m4a")

    elif opt.dataset_name == 'grid':
        # target
        fake_root = '/u/lchen63/Project/face_tracking_detection/eccv2020/sample/grid_test'
        opt.tgt_video_path = os.path.join(root, 'align', paths[0], paths[1]+"_crop.mp4")
        # opt.tgt_lmarks_path = os.path.join(root, 'align', paths[0], paths[1]+"_original.npy")
        opt.tgt_lmarks_path = os.path.join(fake_root, '{}__{}_front_diff_rotated.npy'.format(paths[0], paths[1]))
        # opt.tgt_lmarks_path = os.path.join(fake_root, '{}__{}_front_diff_rotated.npy'.format(paths[0], paths[1]))
        opt.tgt_rt_path = os.path.join(root, 'align', paths[0], paths[1]+ '_rt.npy') 
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

    elif opt.dataset_name == 'crema':
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

root = opt.dataroot
if opt.dataset_name == 'grid':
    _file = open(os.path.join(root, 'pickle','test_audio2lmark_grid.pkl'), "rb")
elif opt.dataset_name == 'crema':
    _file = open(os.path.join(root, 'pickle','train_lmark2img.pkl'), "rb")
elif opt.dataset_name == 'lrw':
    _file = open(os.path.join(root, 'pickle','test3_lmark2img.pkl'), "rb")
elif opt.dataset_name == 'lrs':
    _file = open(os.path.join(root, 'pickle','test2_lmark2img.pkl'), "rb")
else:
    _file = open(os.path.join(root, 'pickle','test_lmark2img.pkl'), "rb")
pickle_data_store = pkl.load(_file)
_file.close()

if opt.dataset_name == 'crema':
    pickle_data_store = pickle_data_store[int(len(pickle_data_store)*0.8):]
# pickle_data = [['id00081', '2xYrsnvtUWc', '00002'], ['id00081', '2xYrsnvtUWc', '00004'], ['id01000', '0lmrq0quo9M', '00001']]
# pickle_files = get_good_file()

human_ids = list(set([data[0].split('_')[0] for data in pickle_data_store]))
# emotions = list(set([data[0].split('_')[2] for data in pickle_data_store]))
emotions = ['HAP', 'SAD']

for emo in tqdm(emotions):
    if emo == 'NEU':
        continue
    for h_id in tqdm(human_ids):

        ################# reset ################
        opt = TestOptions().parse()

        # for finetune
        pickle_data = [data for data in pickle_data_store if h_id in data[0] and emo in data[0]]
        # pickle_data = [data for data in pickle_data if 's14' in data[0]]
        finetune_data = pickle_data
        # finetune_ids = [0,2,4,6,8,10]
        finetune_ids = list(range(0, len(pickle_data), 2))

        if len(pickle_data) < 2:
            continue

        ### setup models
        model = create_model(opt)
        model.eval()

        # preprocess
        save_name = opt.name
        if opt.dataset_name == 'lrs':
            save_name = 'lrs'
        if opt.dataset_name == 'lrw':
            save_name = 'lrw'
        # save_root = os.path.join('evaluation_store', save_name, '{}_shot_test'.format(opt.n_shot), 'epoch_{}'.format(opt.which_epoch))
        # save_root = os.path.join('evaluation_store', save_name, '{}_shot_epoch_{}'.format(opt.n_shot, opt.which_epoch))
        save_root = os.path.join('evaluation_finetune', save_name, 'finetune_{}'.format(opt.finetune_shot), '{}_shot_epoch_{}'.format(opt.n_shot, opt.which_epoch))
        # pick_ids = np.random.choice(list(range(len(pickle_data))), size=opt.how_many)
        # save_root = os.path.join('audio_result', save_name, '{}_shot_epoch_{}'.format(opt.n_shot, opt.which_epoch))
        end = int(len(pickle_data))
        # pick_ids = range(1, end-5, (end)//opt.how_many)
        # pick_ids = range(0, end-5, end//opt.how_many)
        pick_ids = range(0, end)
        # pick_ids = [100]
        # pick_ids = range(0, opt.how_many)
        # pick_ids = range(0, len(pickle_data))
        # pick_files = ['s23', 's29']
        # pick_files = ['s29']
        # pick_files = ['1075_IWL_FEA_XX_', '1090_IOM_FEA_XX_', '1088_ITH_HAP_XX_', '1091_IWL_NEU_XX_', \
        #               '1085_TIE_HAP_XX_', '1075_TIE_HAP_XX_', '1077_WSI_FEA_XX__', '1089_IWL_ANG_XX_']
        # pick_files = np.asarray([['id00081', '2xYrsnvtUWc', '00002'], ['id00081', '2xYrsnvtUWc', '00004'], ['id01000', '0lmrq0quo9M', '00001']])

        # pick_files = np.asarray([['id04094', '2sjuXzB2I1M', '00025'], \
        #                ['id00081', '2xYrsnvtUWc', '00002'], \
        #                ['id00081', '2xYrsnvtUWc', '00004'], \
        #                ['id01000', '0lmrq0quo9M', '00001']])
        # pick_ids = range(0, len(pickle_data))

        test_ref_id = opt.ref_img_id
        test_shot = opt.n_shot
        # finetune 
        opt.ref_img_id = ""
        opt.n_shot = opt.finetune_shot
        step = 64 // opt.finetune_shot
        for i in range(0, opt.finetune_shot):
            opt.ref_img_id += '{},'.format(i * step)
        opt.ref_img_id = opt.ref_img_id[:-1]

        refs, warps = [], []
        for f_id in finetune_ids: 
            get_param(root, finetune_data, f_id, opt)
            data_loader = CreateDataLoader(opt)
            dataset = data_loader.load_data()
            data = next(iter(dataset))
            ref_labels, ref_images = data['ref_label'], data['ref_image']
            warping_ref_lmark, warping_ref = data['warping_ref_lmark'], data['warping_ref']
            refs.append([ref_labels, ref_images])
            warps.append([warping_ref_lmark, warping_ref])


        ref_label_list, ref_image_list = [], []
        tgt_label_list, tgt_image_list = [], []
        warp_label_list, warp_image_list = [], []
        total_iterations = max(30, opt.finetune_shot*10)
        # total_iterations = 0
        for ref, warp in zip(refs, warps):
            ref_labels, ref_images = ref
            warping_ref_lmark, warping_ref = warp
            tgt_label_list.append(ref_labels)
            tgt_image_list.append(ref_images)
            ref_label_list.append(ref_labels[:,0].unsqueeze(1))
            ref_image_list.append(ref_images[:,0].unsqueeze(1))
            warp_label_list.append(warping_ref_lmark)
            warp_image_list.append(warping_ref)

        model.finetune_call_multi(tgt_label_list, tgt_image_list, \
                        ref_label_list, ref_image_list, \
                        warp_label_list, warp_image_list, iterations=total_iterations)

        img_path = data['path']
        img_id = "finetune_{}_{}_{}".format(img_path[0].split('/')[-3], img_path[0].split('/')[-2], img_path[0].split('/')[-1][:-4])
        img_dir = os.path.join(save_root,  img_id)

        if not os.path.exists(img_dir):
                os.makedirs(img_dir)
        for ref_img_id in range(ref_images.shape[1]):
            ref_img = util.tensor2im(ref_images[0, ref_img_id])
            ref_label = util.tensor2im(ref_labels[0, ref_img_id])
            save_img = np.hstack([ref_img, ref_label])
            save_img = Image.fromarray(save_img)
            save_img.save(os.path.join(img_dir, 'ref_{}.png').format(ref_img_id))

        # test
        opt.ref_img_id = test_ref_id
        opt.n_shot = test_shot
        count = 0
        pick_ids = list(range(1, len(pickle_data), 2))
        # pick_ids = [0,1,2]
        for pick_id in tqdm(pick_ids):
            paths = pickle_data[pick_id]
            # if '{}_{}_{}_aligned'.format(paths[0], paths[1], paths[2]) not in pickle_files:
            #     continue
            # pdb.set_trace()
            # if 'test_{}_{}_crop'.format(paths[0], paths[1][:5]) not in pickle_files:
            #     continue

            count += 1
            if count == 5:
                break

            print('process {} ...'.format(pick_id))
            audio_tgt_path = get_param(root, pickle_data, pick_id, opt)

            # if paths[0] not in pick_files:
            #     continue
            # if paths[0][:-10] not in pick_files:
            #     continue
            # if not((paths[1] in pick_files[:, 1]) and (paths[0] in pick_files[:, 0]) and (paths[2] in pick_files[:, 2])):
            #     continue
            # count += 1
            # if count > opt.how_many:
            #     break


            ### setup dataset
            data_loader = CreateDataLoader(opt)
            dataset = data_loader.load_data()

            # if len(dataset) <= 70:
            #     continue

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
                img_name = "%05d.jpg"%data['index'][0]

                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                image_pil = Image.fromarray(compare_image)
                image_pil.save(os.path.join(img_dir, img_name))

                # save for test
                # test_syn_image = util.tensor2im(synthesized_image)
                # img_test_dir = os.path.join(save_root, 'test')
                # if not os.path.exists(img_test_dir):
                #     os.makedirs(img_test_dir)
                # image_pil = Image.fromarray(test_syn_image)
                # image_pil.save(os.path.join(img_test_dir, img_name))

                # save reference
                if i == 0:
                    if not os.path.exists(os.path.join(img_dir, 'reference')):
                        os.makedirs(os.path.join(img_dir, 'reference'))
                    for ref_img_id in range(data['ref_image'].shape[1]):
                        ref_img = util.tensor2im(data['ref_image'][0, ref_img_id])
                        ref_img = Image.fromarray(ref_img)
                        ref_img.save(os.path.join(img_dir, 'reference', 'ref_{}.png').format(ref_img_id))

                # save for evaluation
                # if opt.evaluate:
                #     if not os.path.exists(os.path.join(img_dir, 'real')):
                #         os.makedirs(os.path.join(img_dir, 'real'))
                #     img_path = os.path.join(img_dir, 'real', '{}_{}_image.png'.format(data['target_id'][0], 'real'))
                #     image_pil = Image.fromarray(tgt_image)
                #     image_pil.save(img_path)

                #     if not os.path.exists(os.path.join(img_dir, 'synthesized')):
                #         os.makedirs(os.path.join(img_dir, 'synthesized'))
                #     img_path = os.path.join(img_dir, 'synthesized', '{}_{}_image.png'.format(data['target_id'][0], 'synthesized'))
                #     image_pil = Image.fromarray(synthesized_image)
                #     image_pil.save(img_path)

                # print('process image... %s' % img_path)

            # combine into video (save for compare)
            v_n = os.path.join(img_dir, 'test.mp4')
            image_to_video(sample_dir = img_dir, video_name = v_n)
            add_audio(os.path.join(img_dir, 'test.mp4'), audio_tgt_path)
            # combine into video (save for test)
            # v_n = os.path.join(img_test_dir, '{}_fake.mp4'.format(img_id))
            # image_to_video(sample_dir = img_test_dir, video_name = v_n)
            # add_audio(v_n, audio_tgt_path)
            # for f in os.listdir(img_test_dir):
            #     if f.split('.')[1] != "mp4":
            #         os.remove(os.path.join(img_test_dir, f))