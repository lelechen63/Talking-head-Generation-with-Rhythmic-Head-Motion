
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
    
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.jpg -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 
    #ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
    print (command)
    os.system(command)

def set_param_for_grid(root, pickle_data, pick_id, opt):
    if opt.dataset_name == 'grid':
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
        opt.ref_video_path = os.path.join(root, 'align', "s14", "brbl5p_crop.mp4")
        opt.ref_lmarks_path = os.path.join(root, 'align', "s14", "brbl5p_original.npy")
        opt.ref_rt_path = os.path.join(root, 'align', "s14", 'brbl5p_rt.npy') 
        opt.ref_ani_id = None

        audio_tgt_path = os.path.join(root, 'audio', paths[0], paths[1]+".wav")

    return audio_tgt_path

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

    elif opt.dataset_name == 'obama':
        # target
        opt.tgt_video_path = os.path.join(root, 'video', paths[0][:-11]+"_crop2.mp4")
        # opt.tgt_lmarks_path = os.path.join(root, 'video', paths[0][:-11]+"_original2.npy")
        # opt.tgt_rt_path = os.path.join(root, 'video', paths[0][:-11]+"_rt2.npy")
        # opt.tgt_ani_path = os.path.join(root, 'video', paths[0][:-11]+"_ani2.mp4")
        opt.tgt_lmarks_path = "/home/cxu-serve/p1/common/demo/demo_3_3__rotated.npy"
        opt.tgt_rt_path = "/home/cxu-serve/p1/common/demo/00025_aligned_rt.npy"
        opt.tgt_ani_path = "/home/cxu-serve/p1/common/demo/demo_00025_3_3__ani2.mp4"
        # reference
        ref_paths = paths
        # opt.ref_front_path = os.path.join(root, 'video', paths[0][:-11]+"_front2.npy")
        # opt.ref_video_path = opt.tgt_video_path
        # opt.ref_lmarks_path = opt.tgt_lmarks_path
        # opt.ref_rt_path = opt.tgt_rt_path
        # opt.ref_ani_id = int(paths[1])
        opt.ref_front_path = os.path.join(root, 'video', paths[0][:-11]+"_front2.npy")
        opt.ref_video_path = os.path.join(root, 'video', paths[0][:-11]+"_crop2.mp4")
        opt.ref_lmarks_path = os.path.join(root, 'video', paths[0][:-11]+"_original2.npy")
        opt.ref_rt_path = os.path.join(root, 'video', paths[0][:-11]+"_rt2.npy")
        opt.ref_ani_id = int(paths[1])

        audio_tgt_path = os.path.join(root, 'video', paths[0].split('__')[0]+".wav")

    elif opt.dataset_name == 'obama_front':
        # target
        opt.tgt_video_path = os.path.join(root, 'video', paths[0][:-11]+"_crop2.mp4")
        opt.tgt_lmarks_path = os.path.join(root, 'video', paths[0][:-11]+"_front2.npy")
        opt.tgt_rt_path = os.path.join(root, 'video', paths[0][:-11]+"_rt2.npy")
        opt.tgt_ani_path = os.path.join(root, 'video', paths[0][:-11]+"_ani2.mp4")
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, 'video', paths[0][:-11]+"_front2.npy")
        opt.ref_video_path = os.path.join(root, 'video', paths[0][:-11]+"_crop2.mp4")
        opt.ref_lmarks_path = os.path.join(root, 'video', paths[0][:-11]+"_original2.npy")
        opt.ref_rt_path = os.path.join(root, 'video', paths[0][:-11]+"_rt2.npy")
        opt.ref_ani_id = int(paths[1])

        audio_tgt_path = os.path.join(root, 'video', paths[0].split('__')[0]+".wav")


    elif opt.dataset_name == 'obama_fake':
        # target
        opt.tgt_video_path = os.path.join(root, 'video', paths[0][:-11]+"_crop2.mp4")
        opt.tgt_lmarks_path = "/home/cxu-serve/p1/common/demo/demo_3_3__rotated.npy"
        opt.tgt_rt_path = "/home/cxu-serve/p1/common/demo/3_3__rt2.npy"
        opt.tgt_ani_path = os.path.join("/home/cxu-serve/p1/common/demo/demo_3_3__ani2.mp4")
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, 'video', paths[0][:-11]+"_front2.npy")
        opt.ref_video_path = os.path.join(root, 'video', paths[0][:-11]+"_crop2.mp4")
        opt.ref_lmarks_path = os.path.join(root, 'video', paths[0][:-11]+"_original2.npy")
        opt.ref_rt_path = os.path.join(root, 'video', paths[0][:-11]+"_rt2.npy")
        opt.ref_ani_id = int(paths[1])

        opt.n_shot = 128
        opt.ref_img_id = ''
        for i in range(0, 128 * 3, 3):
            opt.ref_img_id += '{},'.format(i)
        opt.ref_img_id = opt.ref_img_id[:-1]

        audio_tgt_path = os.path.join("/home/cxu-serve/p1/common/demo/demo.wav")

    elif opt.dataset_name == 'other_fake':
        # target
        opt.tgt_video_path = "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id04094/2sjuXzB2I1M/00025_aligned.mp4"
        opt.tgt_lmarks_path = "/home/cxu-serve/p1/common/demo/demo_3_3__front.npy"
        opt.tgt_rt_path = "/home/cxu-serve/p1/common/demo/3_3__rt2.npy"
        opt.tgt_ani_path = "/home/cxu-serve/p1/common/demo/demo_3_3__ani2.mp4"
        # reference
        ref_paths = paths
        opt.ref_front_path = "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id04094/2sjuXzB2I1M/00025_aligned_front.npy"
        opt.ref_video_path = "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id04094/2sjuXzB2I1M/00025_aligned.mp4"
        opt.ref_lmarks_path = "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id04094/2sjuXzB2I1M/00025_aligned.npy"
        opt.ref_rt_path = "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id04094/2sjuXzB2I1M/00025_aligned_rt.npy"
        opt.ref_ani_id = 243

        # opt.n_shot = 128
        # opt.ref_img_id = ''
        # for i in range(0, 128 * 3, 3):
        #     opt.ref_img_id += '{},'.format(i)
        # opt.ref_img_id = opt.ref_img_id[:-1]

        audio_tgt_path = os.path.join("/home/cxu-serve/p1/common/demo/demo.wav")

    return audio_tgt_path

def sel_data(pickle_data, opt):
    # option 1
    # finetune_files = ['KuWXfjyGhk0_00001']
    # pickle_files = ['KuWXfjyGhk0_00001', 'Mt0PiXLvYlU_00011', 'Ip2SQa50uBI_00001', 'Sa27SUR0Mlo_00002']

    # finetune_data = [data for data in pickle_data if 'align_{}_{}_crop'.format(data[0], data[1]) in finetune_files]
    # pickle_data = [data for data in pickle_data if 'align_{}_{}_crop'.format(data[0], data[1]) in pickle_files]

    # finetune_ids = [0]
    # end = len(pickle_data)
    # pick_ids = range(0, end)

    # option 2
    finetune_data = [data for data in pickle_data if ('2_1' in data[0] or '2_2' in data[0]) and '11' not in data[0]]
    pickle_data = [data for data in pickle_data if '3_3' in data[0]]

    finetune_ids = [0]
    pickle_ids = [0]

    return finetune_data, finetune_ids, pickle_data, pickle_ids


opt = TestOptions().parse()

# preprocess
save_name = opt.name
if opt.dataset_name == 'lrs':
    save_name = 'lrs'
if opt.dataset_name == 'lrw':
    save_name = 'lrw'
if opt.dataset_name == 'obama_fake':
    save_name = 'obama_fake'
if opt.dataset_name == 'obama_front':
    save_name = 'obama_front'

save_root = os.path.join('test', save_name, 'finetune_front_{}'.format(opt.finetune_shot), '{}_shot_epoch_{}'.format(opt.n_shot, opt.which_epoch))

### setup models
model = create_model(opt)
model.eval()

# setup dataset
root = opt.dataroot
if opt.dataset_name == 'grid' or opt.dataset_name == 'grid_diff':
    _file = open(os.path.join(root, 'pickle','test_audio2lmark_grid.pkl'), "rb")
elif opt.dataset_name == 'crema':
    _file = open(os.path.join(root, 'pickle','train_lmark2img.pkl'), "rb")
elif opt.dataset_name == 'lrw':
    _file = open(os.path.join(root, 'pickle','test3_lmark2img.pkl'), "rb")
elif opt.dataset_name == 'lrs':
    _file = open(os.path.join(root, 'pickle','test2_lmark2img.pkl'), "rb")
elif 'obama' in opt.dataset_name:
    _file = open(os.path.join(root, 'pickle', 'train_lmark2img.pkl'), 'rb')
else:
    _file = open(os.path.join(root, 'pickle','test_lmark2img.pkl'), "rb")
pickle_data = pkl.load(_file)
_file.close()

if opt.dataset_name == 'crema':
    pickle_data = pickle_data[int(len(pickle_data)*0.8):]

# for finetune
finetune_data, finetune_ids, pickle_data, pick_ids = sel_data(pickle_data, opt)

test_ref_id = opt.ref_img_id
test_shot = opt.n_shot
# finetune 
opt.ref_img_id = ""
opt.n_shot = opt.finetune_shot
step = 64 // opt.finetune_shot
for i in range(0, opt.finetune_shot):
    opt.ref_img_id += '{},'.format(i * step)
opt.ref_img_id = opt.ref_img_id[:-1]

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
total_iterations = 0
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
img_id = "finetune_{}_{}_{}".format(img_path[0].split('/')[-3], img_path[0].split('/')[-2], img_path[0].split('/')[-1][:-4])
img_dir = os.path.join(save_root,  img_id)

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

# model.save_networks('finetune')

# test
opt.ref_img_id = test_ref_id
opt.n_shot = test_shot
count = 0
# pick_ids = [3,5,7,9,11,12,13,14,15,16]
# pick_ids = [17,18,19,20,21,22,23,24,25,26]
# pick_ids = [0,1]
pick_ids = [0]
for pick_id in tqdm(pick_ids):
    paths = pickle_data[pick_id]
    # if '{}_{}_{}_aligned'.format(paths[0], paths[1], paths[2]) not in pickle_files:
    #     continue
    # pdb.set_trace()
    # if 'test_{}_{}_crop'.format(paths[0], paths[1][:5]) not in pickle_files:
    #     continue
    # if 'align_{}_{}_crop'.format(paths[0], paths[1]) not in pickle_files:
    #     continue
    # if '{}_{}'.format(paths[0], paths[1]) not in pickle_files:
    #     continue

    # count += 1
    # if count == 20:
    #     break

    print('process {} ...'.format(pick_id))
    audio_tgt_path = get_param(root, pickle_data, pick_id, opt)
    # audio_tgt_path = set_param_for_grid(root, pickle_data, pick_id, opt)

    # if 's14' in paths[0]:
    #     audio_tgt_path = get_param(root, pickle_data, pick_id, opt)
    # elif 's15' in paths[0]:
    #     audio_tgt_path = set_param_for_grid(root, pickle_data, pick_id, opt)

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
        if i >= 500: break
        if i >= len(dataset): break
        img_path = data['path']
        if not opt.warp_ani:
            data.update({'ani_image':None, 'ani_lmark':None, 'cropped_images':None, 'cropped_lmarks':None })
        if "warping_ref" not in data:
            data.update({'warping_ref': data['ref_image'][:, :1], 'warping_ref_lmark': data['ref_label'][:, :1]})
        # data.update({'warping_ref': data['ref_image'][:, :1], 'warping_ref_lmark': data['ref_label'][:, :1]})

        # pdb.set_trace()

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