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

# import soft_renderer as sr
# import soft_renderer.cuda.create_texture_image as create_texture_image_cuda
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import os


def get_3d(bbb):
   
    # ---- init PRN
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)

    # ------------- load data
    # frame_id = "test_video/id00419/3U0abyjM2Po/00024"
    # mesh_file = os.path.join(root, frame_id + ".obj") 
    # rt_file = os.path.join(root, frame_id + "_sRT.npy")
    # image_path 

    # _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    # data = pickle._Unpickler(_file)
    # data.encoding = 'latin1'
    # data = data.load()
    _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    data = pickle.load(_file)
    _file.close()
    gg = len(data)
    print (len(data))
    data = data[int(gg * 0.1 *bbb ): int(gg * 0.1 * (bbb + 1) ) ]
    for kk ,item in enumerate(data) :
        print (kk)
        
        target_id = item[-1]
        video_path = os.path.join(root, 'unzip', item[0] + '.mp4')        
        if not os.path.exists(video_path):
            print (video_path) 
            print ('+++++')
            continue
        if os.path.exists(video_path[:-4] + '.obj'):
            print ('-----')
            continue
        cap = cv2.VideoCapture(video_path)
        for i in range(target_id):
            ret, frame = cap.read()
        ret, target_frame = cap.read()
        cv2.imwrite(video_path[:-4] + '_%05d.png'%target_id,target_frame)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)



        image = target_frame
        # read image
        [h, w, c] = image.shape
        
        pos = prn.process(image) # use dlib to detect face
        
        image = image/255.
        if pos is None:
            continue
        

        # landmark
        kpt = prn.get_landmarks(pos)
        kpt[:,1] = 224 - kpt[:,1]

        np.save(video_path[:-4] + '_prnet.npy', kpt)
        # 3D vertices
        vertices = prn.get_vertices(pos)
        # save_vertices, p = frontalize(vertices)
        # np.save(video_path[:-4] + '_p.npy', p) 
        # if os.path.exists(video_path[:-4] + '.obj'):
        #     continue
        save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        
        # corresponding colors
        colors = prn.get_colors(image, vertices)
        
        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        write_obj_with_colors(video_path[:-4] + '_original.obj', save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

        
        # print (video_path)
        # break

def get_3d_folder(pkl): # the first cell is video path the last cell is the key frame nnuumber
    
    # ---- init PRN
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)
    _file = open(pkl, "rb")
    data = pickle.load(_file)
    _file.close()
    gg = len(data)
    print (len(data))
    # data = data[int(gg * 0.1 *bbb ): int(gg * 0.1 * (bbb + 1) ) ]
    for kk ,item in enumerate(data) :
        print (kk)
        
        target_id = item[-1]
        video_path = os.path.join(root, 'unzip', item[0] )        
        if not os.path.exists(video_path):
            print (video_path) 
            print ('+++++')
            continue
        if os.path.exists(video_path[:-4] + '.obj'):
            print ('-----')
            continue
        cap = cv2.VideoCapture(video_path)
        for i in range(target_id):
            ret, frame = cap.read()
        ret, target_frame = cap.read()
        cv2.imwrite(video_path[:-4] + '_%05d.png'%target_id,target_frame)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)



        image = target_frame
        # read image
        [h, w, c] = image.shape
        
        pos = prn.process(image) # use dlib to detect face
        
        image = image/255.
        if pos is None:
            print ('there is no pos!!! WRONG!')
            continue
        

        # landmark
        kpt = prn.get_landmarks(pos)
        kpt[:,1] = 224 - kpt[:,1]

        np.save(video_path[:-4] + '_prnet.npy', kpt)
        # 3D vertices
        vertices = prn.get_vertices(pos)
        # save_vertices, p = frontalize(vertices)
        # np.save(video_path[:-4] + '_p.npy', p) 
        # if os.path.exists(video_path[:-4] + '.obj'):
        #     continue
        save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        
        # corresponding colors
        colors = prn.get_colors(image, vertices)
        
        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        write_obj_with_colors(video_path[:-4] + '_original.obj', save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

        
        # print (video_path)
        # break


def get_3d_pkl_lrs(pkl, root , bbb): # the first cell is video path the last cell is the key frame nnuumber
    
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)
    _file = open(pkl, "rb")
    data = pickle.load(_file)
    _file.close()
    gg = len(data)
    
    data = data[int(gg * 0.5 *bbb ): int(gg * 0.5 * (bbb + 1) ) ]
    for kk ,item in enumerate(data) :
        print (kk)
        print (item)
        target_id = item[-1]

        img_path = os.path.join(root, 'test', item[0] , item[1][:5] + '_%05d.png'%target_id  )
        obj_path = os.path.join(root,  'test', item[0] , item[1][:5] + '_original.obj')
        # if os.path.exists(obj_path):
        #     print ('-----')
        #     continue
        # img_path =  os.path.join(root, 'VideoFlash',  item[0][:-10] + '_%05d.png'%target_id  )
        print (img_path)
        target_frame = cv2.imread(img_path)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

        image = target_frame
        # read image
        [h, w, c] = image.shape
        
        pos = prn.process(image) # use dlib to detect face
        
        image = image/255.
        if pos is None:
            print ('==+++')
            continue
        
        # landmark
        kpt = prn.get_landmarks(pos)
        kpt[:,1] = h - kpt[:,1]

        np.save(os.path.join(root,  'test', item[0] , item[1][:5] + '_prnet.npy'), kpt)
        # 3D vertices
        vertices = prn.get_vertices(pos)
        # save_vertices, p = frontalize(vertices)
        # np.save(video_path[:-4] + '_p.npy', p) 
        # if os.path.exists(video_path[:-4] + '.obj'):
        #     continue
        save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        
        # corresponding colors
        colors = prn.get_colors(image, vertices)
        
        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        write_obj_with_colors( obj_path, save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

        
        # print (video_path)
        # break

def get_3d_pkl_crema(pkl, root ,bbb = 0): # the first cell is video path the last cell is the key frame nnuumber
    
    # ---- init PRN
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)
    _file = open(pkl, "rb")
    data = pickle.load(_file)
    _file.close()
    gg = len(data)
    
    data = data[int(gg * 0.5 *( bbb) ): int(gg * 0.5 * (bbb + 1) ) ]
    for kk ,item in enumerate(data) :
        print (kk)
        print (item)
        target_id = item[-2]

        img_path =  os.path.join(root, 'VideoFlash',  item[0][:-10] + '_%05d.png'%target_id  )
        print (img_path)
        target_frame = cv2.imread(img_path)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

        image = target_frame
        # read image
        [h, w, c] = image.shape
        
        pos = prn.process(image) # use dlib to detect face
        
        image = image/255.
        if pos is None:
            continue
        

        # landmark
        kpt = prn.get_landmarks(pos)
        kpt[:,1] = h - kpt[:,1]

        np.save(os.path.join(root, 'VideoFlash', item[0][:-10] + '_prnet.npy'), kpt)
        # 3D vertices
        vertices = prn.get_vertices(pos)
    
        save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        
        # corresponding colors
        colors = prn.get_colors(image, vertices)
        
        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        write_obj_with_colors(os.path.join(root, 'VideoFlash', item[0][:-10]  + '_original.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

        
        # print (video_path)
        # break


def get_3d_pkl_obama(pkl , root ,bbb = 0): # the first cell is video path the last cell is the key frame nnuumber
    # root = 
    # ---- init PRN
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)
    _file = open(pkl, "rb")
    data = pickle.load(_file)
    _file.close()
    gg = len(data)
    
    data = data[int(gg * 0.2 *( bbb) ): int(gg * 0.2 * (bbb + 1) ) ]
    for kk ,item in enumerate(data) :
        print (kk)
        print (item)
        target_id = item[-1]

        img_path =  os.path.join(root, 'video',  item[0][:-11] + '_%05d_2.png'%target_id  )
        print (img_path)
        target_frame = cv2.imread(img_path)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

        image = target_frame
        # read image
        [h, w, c] = image.shape
        
        pos = prn.process(image) # use dlib to detect face
        
        image = image/255.
        if pos is None:
            continue
        

        # landmark
        kpt = prn.get_landmarks(pos)
        kpt[:,1] = h - kpt[:,1]

        np.save(os.path.join(root, 'video', item[0][:-11] + '_prnet2.npy'), kpt)
        # 3D vertices
        vertices = prn.get_vertices(pos)
    
        save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        
        # corresponding colors
        colors = prn.get_colors(image, vertices)
        
        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        write_obj_with_colors(os.path.join(root, 'video', item[0][:-11]  + '_original2.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

        
        # print (video_path)
        # break



def get_3d_single_video( img_path, with_frame_num=False): # you need the image path of the most visible frame.
    # root = 
    # ---- init PRN
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)
    # _file = open(pkl, "rb")
    # data = pickle.load(_file)
    # _file.close()
    # gg = len(data)
    
    # data = data[int(gg * 0.2 *( bbb) ): int(gg * 0.2 * (bbb + 1) ) ]
    # for kk ,item in enumerate(data) :
       
       
    print (img_path)
    target_frame = cv2.imread(img_path)
    target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

    image = target_frame
    # read image
    [h, w, c] = image.shape
    
    pos = prn.process(image) # use dlib to detect face
    
    image = image/255.
    if pos is None:
        print ('No pos')
    

    # landmark
    kpt = prn.get_landmarks(pos)
    kpt[:,1] = h - kpt[:,1]
    if with_frame_num:
       np.save(img_path[:-11] + '__prnet.npy', kpt)
    else:
      np.save(img_path[:-4] + '__prnet.npy', kpt)
    # 3D vertices
    vertices = prn.get_vertices(pos)

    save_vertices = vertices.copy()
    save_vertices[:,1] = h - 1 - save_vertices[:,1]

    
    # corresponding colors
    colors = prn.get_colors(image, vertices)
    
    # print (colors.shape)
    # print ('=========')
    # cv2.imwrite('./mask.png', colors * 255)
    if with_frame_num:
       write_obj_with_colors(img_path[:-11]  + '__original.obj', save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)
    else:
       write_obj_with_colors(img_path[:-4]  + '__original.obj', save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)
        
        # print (video_path)
        # break

def get_3d_pkl_lrw(pkl, root ,bbb = 0): # the first cell is video path the last cell is the key frame nnuumber
    
    # ---- init PRN
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)
    _file = open(pkl, "rb")
    data = pickle.load(_file)
    _file.close()
    gg = len(data)
    
    data = data[int(gg * 1  *( bbb) ): int(gg * 1  * (bbb + 1) ) ]
    for kk ,item in enumerate(data) :
        print (kk)
        print (item)
        if os.path.exists(item[0] +  '_original.obj'):
            continue
        target_id = item[-1]

        img_path =  item[0] + '_%05d.png'%target_id 
        print (img_path)
        target_frame = cv2.imread(img_path)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

        image = target_frame
        # read image
        [h, w, c] = image.shape
        
        pos = prn.process(image) # use dlib to detect face
        
        image = image/255.
        if pos is None:
            print  ('+++++')
            continue
        

        # landmark
        kpt = prn.get_landmarks(pos)
        kpt[:,1] = h - kpt[:,1]

        np.save(item[0] +  '_prnet.npy', kpt)
        # 3D vertices
        vertices = prn.get_vertices(pos)
    
        save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        
        # corresponding colors
        colors = prn.get_colors(image, vertices)
        
        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        write_obj_with_colors(item[0] +  '_original.obj', save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

        


def get_3d_single(video_path = None, target_id =None,img_path =None):
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)
    if video_path != None:
        if not os.path.exists(video_path):
            print (video_path) 
            print ('+++++')
        if os.path.exists(video_path[:-4] + '.obj'):
            print ('-----')
        cap = cv2.VideoCapture(video_path)
        for i in range(target_id):
            ret, frame = cap.read()
        ret, target_frame = cap.read()
        cv2.imwrite(video_path[:-4] + '_%05d.png'%target_id,target_frame)
    elif img_path != None:
        target_frame = cv2.imread(img_path)
    target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)


    image = target_frame
    # read image
    [h, w, c] = image.shape
    
    pos = prn.process(image) # use dlib to detect face
    
    image = image/255.
    # landmark
    kpt = prn.get_landmarks(pos)
    kpt[:,1] = h - kpt[:,1]
    if video_path != None:
        np.save(video_path[:-4] + '_prnet.npy', kpt)
    else:
        np.save(img_path[:-4] + '_prnet.npy', kpt)
    # 3D vertices
    vertices = prn.get_vertices(pos)
    #
    save_vertices = vertices.copy()
    save_vertices[:,1] = h - 1 - save_vertices[:,1]
    # corresponding colors
    colors = prn.get_colors(image, vertices)
    
    if video_path != None:
        write_obj_with_colors(video_path[:-4] + '_original.obj', save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)
        print ('The generated 3d mesh model is stored in ' + video_path[:-4] + '_original.obj')
    else:
        write_obj_with_colors(img_path[:-4] + '_original.obj', save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)
        print ('The generated 3d mesh model is stored in ' + img_path[:-4] + '_original.obj')

    

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
    parser.add_argument("--with_frame_num",
                        action='store_true')
      
    return parser.parse_args()


# get_3d(config.b)    

# get_3d_single( img_path =config.img_path)    

# get_3d_single( video_path= config.v_path, target_id=config.target_id)    
# root = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4'
# root = '/home/cxu-serve/p1/common/Obama'
# bbb = config.b
# get_3d_pkl_lrs(os.path.join( root, 'pickle','test2_lmark2img.pkl'), root ,bbb)

# os.environ['CUDA_VISIBLE_DEVICES'] = str(config.b)
# get_3d_pkl_obama(os.path.join( root,  'pickle', 'train_lmark2img.pkl'),root,bbb)
# get_3d_pkl_lrw(os.path.join( root,  'pickle', 'test2_lmark2img.pkl'),root,bbb)

def main():
    config = parse_args()
    print ('NOTE, you need to enter the image path that obtained in get_front function!!!! We compute the 3D model based on that frame')
    with_frame_num = config.with_frame_num
    get_3d_single_video(img_path = config.img_path, with_frame_num=with_frame_num)

main()
# get_3d_single(img_path= '/home/cxu-serve/p1/common/demo/self2_crop.png')
