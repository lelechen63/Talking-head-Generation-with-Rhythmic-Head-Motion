import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import cv2
from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture
import pickle

import soft_renderer as sr
import soft_renderer.cuda.create_texture_image as create_texture_image_cuda
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import os

root = '/mnt/Data/lchen63/voxceleb/'


def get_3d(bbb):
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU number, -1 for CPU
    prn = PRN(is_dlib=True)

    # ------------- load data
    # frame_id = "test_video/id00419/3U0abyjM2Po/00024"
    # mesh_file = os.path.join(root, frame_id + ".obj")
    # rt_file = os.path.join(root, frame_id + "_sRT.npy")
    # image_path

    # _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    # data = pickle._Unpickler(_file)
    # data.encoding = 'latin1'
    # data = data.load()
    _file = open(os.path.join(root, 'txt', "front_rt.pkl"), "rb")
    data = pickle.load(_file)
    _file.close()
    gg = len(data)
    print(len(data))
    data = data[int(gg * 0.1 * bbb): int(gg * 0.1 * (bbb + 1))]
    for kk, item in enumerate(data):
        print(kk)

        target_id = item[-1]
        video_path = os.path.join(root, 'unzip', item[0] + '.mp4')
        if not os.path.exists(video_path):
            print(video_path)
            print('+++++')
            continue
        if os.path.exists(video_path[:-4] + '.obj'):
            print('-----')
            continue
        cap = cv2.VideoCapture(video_path)
        for i in range(target_id):
            ret, frame = cap.read()
        ret, target_frame = cap.read()
        cv2.imwrite(video_path[:-4] + '_%05d.png' % target_id, target_frame)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

        image = target_frame
        # read image
        [h, w, c] = image.shape

        pos = prn.process(image)  # use dlib to detect face

        image = image / 255.
        if pos is None:
            continue

        # landmark
        kpt = prn.get_landmarks(pos)
        kpt[:, 1] = 224 - kpt[:, 1]

        np.save(video_path[:-4] + '_prnet.npy', kpt)
        # 3D vertices
        vertices = prn.get_vertices(pos)
        # save_vertices, p = frontalize(vertices)
        # np.save(video_path[:-4] + '_p.npy', p)
        # if os.path.exists(video_path[:-4] + '.obj'):
        #     continue
        save_vertices = vertices.copy()
        save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

        # corresponding colors
        colors = prn.get_colors(image, vertices)

        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        write_obj_with_colors(video_path[:-4] + '_original.obj', save_vertices, prn.triangles,
                              colors)  # save 3d face(can open with meshlab)

        # print (video_path)
        # break


def get_3d_folder(pkl):  # the first cell is video path the last cell is the key frame nnuumber

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU number, -1 for CPU
    prn = PRN(is_dlib=True)

    # ------------- load data
    # frame_id = "test_video/id00419/3U0abyjM2Po/00024"
    # mesh_file = os.path.join(root, frame_id + ".obj")
    # rt_file = os.path.join(root, frame_id + "_sRT.npy")
    # image_path

    # _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    # data = pickle._Unpickler(_file)
    # data.encoding = 'latin1'
    # data = data.load()
    _file = open(pkl, "rb")
    data = pickle.load(_file)
    _file.close()
    gg = len(data)
    print(len(data))
    # data = data[int(gg * 0.1 *bbb ): int(gg * 0.1 * (bbb + 1) ) ]
    for kk, item in enumerate(data):
        print(kk)

        target_id = item[-1]
        video_path = os.path.join(root, 'unzip', item[0])
        if not os.path.exists(video_path):
            print(video_path)
            print('+++++')
            continue
        if os.path.exists(video_path[:-4] + '.obj'):
            print('-----')
            continue
        cap = cv2.VideoCapture(video_path)
        for i in range(target_id):
            ret, frame = cap.read()
        ret, target_frame = cap.read()
        cv2.imwrite(video_path[:-4] + '_%05d.png' % target_id, target_frame)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

        image = target_frame
        # read image
        [h, w, c] = image.shape

        pos = prn.process(image)  # use dlib to detect face

        image = image / 255.
        if pos is None:
            continue

        # landmark
        kpt = prn.get_landmarks(pos)
        kpt[:, 1] = 224 - kpt[:, 1]

        np.save(video_path[:-4] + '_prnet.npy', kpt)
        # 3D vertices
        vertices = prn.get_vertices(pos)
        # save_vertices, p = frontalize(vertices)
        # np.save(video_path[:-4] + '_p.npy', p)
        # if os.path.exists(video_path[:-4] + '.obj'):
        #     continue
        save_vertices = vertices.copy()
        save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

        # corresponding colors
        colors = prn.get_colors(image, vertices)

        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        write_obj_with_colors(video_path[:-4] + '_original.obj', save_vertices, prn.triangles,
                              colors)  # save 3d face(can open with meshlab)

        # print (video_path)
        # break


import cv2

def is_cv3(or_better=False):
    # grab the OpenCV major version number
    major = get_opencv_major_version()

    # check to see if we are using *at least* OpenCV 3
    if or_better:
        return major >= 3

    # otherwise we want to check for *strictly* OpenCV 3
    return major == 3

def get_opencv_major_version(lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib

    # return the major version number
    return int(lib.__version__.split(".")[0])

def count_frames(path, override=False):
    # grab a pointer to the video file and initialize the total
    # number of frames read
    video = cv2.VideoCapture(path)
    total = 0

    # if the override flag is passed in, revert to the manual
    # method of counting frames
    if override:
        total = count_frames_manual(video)
    # otherwise, let's try the fast way first
    else:
        # lets try to determine the number of frames in a video
        # via video properties; this method can be very buggy
        # and might throw an error based on your OpenCV version
        # or may fail entirely based on your which video codecs
        # you have installed
        try:
            # check if we are using OpenCV 3
            if is_cv3():
                total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # otherwise, we are using OpenCV 2.4
            else:
                total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        # uh-oh, we got an error -- revert to counting manually
        except:
            total = count_frames_manual(video)

    # release the video file pointer
    video.release()

    # return the total number of frames in the video
    return total

def count_frames_manual(video):
    # initialize the total number of frames read
    total = 0

    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        # increment the total number of frames read
        total += 1

    # return the total number of frames in the video file
    return total


def get_3d_single(video_path = None, target_id =None,img_path =None,device_id='3'):
    # ---- init PRN
    target_id = count_frames(video_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)
    # if video_path != None:
    #     if not os.path.exists(video_path):
    #         print (video_path)
    #         print ('+++++')
    #     if os.path.exists(video_path[:-4] + '.obj'):
    #         print ('-----')
    #     cap = cv2.VideoCapture(video_path)
    #     for i in range(target_id):
    #         ret, frame = cap.read()
    #     ret, target_frame = cap.read()
    #     cv2.imwrite(video_path[:-4] + '_%05d.png'%target_id,target_frame)
    # elif img_path != None:
    #     target_frame = cv2.imread(img_path)
    # target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

    cap = cv2.VideoCapture(video_path)
    for i in range(target_id):
        ret, frame = cap.read()
        print(target_path[:-4] + '_%05d.png' % i)
        cv2.imwrite(target_path[:-4] + '_%05d.png' % i, frame)
        # tt = cv2.imread(target_path[:-4] + '_%05d.png' % i)
        # target_frame = cv2.cvtColor(cv2.imread(target_path[:-4] + '_%05d.png' % i), cv2.COLOR_BGR2RGB)
        target_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = target_frame

        [h, w, c] = image.shape

        pos = prn.process(image)
        image = image/255

        kpt = prn.get_landmarks(pos)
        kpt[:, 1] = 224 - kpt[:, 1]
        if video_path is not None:
            print(target_path[:-4] + '_%05d' % i + '_prnet.npy')
            np.save(target_path[:-4] + '_%05d' % i + '_prnet.npy', kpt)
        else:
            np.save(img_path[:-4] + '_%05d' % i + '_prnet.npy', kpt)

        vertices = prn.get_vertices(pos)
        # save_vertices, p = frontalize(vertices)
        # np.save(video_path[:-4] + '_p.npy', p)
        # if os.path.exists(video_path[:-4] + '.obj'):
        #     continue
        save_vertices = vertices.copy()
        save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

        # corresponding colors
        colors = prn.get_colors(image, vertices)

        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        if video_path != None:
            write_obj_with_colors(target_path[:-4] + '_%05d' % i + '_original.obj', save_vertices, prn.triangles,
                                  colors)  # save 3d face(can open with meshlab)
        else:
            write_obj_with_colors(img_path[:-4] + '_%05d' % i + '_original.obj', save_vertices, prn.triangles,
                                  colors)  # save 3d face(can open with meshlab)

    # image = target_frame
    # # read image
    # [h, w, c] = image.shape
    #
    # pos = prn.process(image)  # use dlib to detect face
    #
    # image = image / 255.
    # # landmark
    # kpt = prn.get_landmarks(pos)
    # kpt[:, 1] = 224 - kpt[:, 1]
    # if video_path != None:
    #     np.save(video_path[:-4] + '_prnet.npy', kpt)
    # else:
    #     np.save(img_path[:-4] + '_prnet.npy', kpt)
    # # 3D vertices
    # vertices = prn.get_vertices(pos)
    # # save_vertices, p = frontalize(vertices)
    # # np.save(video_path[:-4] + '_p.npy', p)
    # # if os.path.exists(video_path[:-4] + '.obj'):
    # #     continue
    # save_vertices = vertices.copy()
    # save_vertices[:, 1] = h - 1 - save_vertices[:, 1]
    #
    # # corresponding colors
    # colors = prn.get_colors(image, vertices)
    #
    # # print (colors.shape)
    # # print ('=========')
    # # cv2.imwrite('./mask.png', colors * 255)
    # if video_path != None:
    #     write_obj_with_colors(video_path[:-4] + '_original.obj', save_vertices, prn.triangles,
    #                           colors)  # save 3d face(can open with meshlab)
    # else:
    #     write_obj_with_colors(img_path[:-4] + '_original.obj', save_vertices, prn.triangles,
    #                           colors)  # save 3d face(can open with meshlab)


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b",
                        type=int,
                        default=0)
    parser.add_argument("--v_path",
                        type=str,
                        default=None)
    parser.add_argument("--dname",
                        type=str,
                        default='vox_audio')
    parser.add_argument("--target_id",
                        type=int,
                        default=None)
    parser.add_argument("--img_path",
                        type=str,
                        default=None)
    return parser.parse_args()


config = parse_args()

# get_3d(config.b)

# get_3d_single( img_path =config.img_path)

# get_3d_single( video_path= config.v_path, target_id=config.target_id)
# root = '/mnt/Backup/lchen63/demo_videos'
# root = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/speech'
#
# get_3d_folder(os.path.join(root, 'txt', config.dname + '_key_frame.pkl'))

source_path = '/home/cxu-serve/p1/common/RyersonAudioVisual/Video_Speech_Actor/Actor_01/'
audio_path = '/home/cxu-serve/p1/common/RyersonAudioVisual/Audio_Speech_Actors_01-24/Actor_01/'
id = '04'
video_id = '01-01-%s-01-01-01-01' % id
audio_id = '03-01-%s-01-01-01-01' % id
video_path = source_path + video_id + '.mp4'
audio = audio_path + audio_id + '.wav'
target_path = source_path + '3dmesh/%s/' % id + video_id + '.mp4'
print(video_path)
# cropped_video_path = source_path + 'cropped_video/' + video_id + '_cropped.mp4'
# print(cropped_video_path)
# landmark_path = cropped_video_path[:-12] + '_original.npy'
# print(landmark_path)
get_3d_single(video_path, device_id='1')
