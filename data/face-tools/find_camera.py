import sys
import soft_renderer as sr
import imageio
from skimage.transform import warp
from skimage.transform import AffineTransform
import numpy as np
import cv2

import torch
import mmcv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import time
res = 256
import os
import pickle
# root  = '/mnt/Data/lchen63/voxceleb/'

# root  = '/home/cxu-serve/p1/lchen63/voxceleb/'  fucj
import shutil

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b",
                        type=int,
                        default=0)
    parser.add_argument("--root",
                        type=str,
                        default='/home/cxu-serve/p1/lchen63/voxceleb/oppo/')

    parser.add_argument("--front_img_path",
                        type=str,
                        default='')
    parser.add_argument("--front_lmark_path",
                        type=str,
                        default='')
    
    parser.add_argument("--prnet_lmark_path",
                        type=str,
                        default='')
    parser.add_argument("--ref_lmark_path",
                        type=str,
                        default='')
    parser.add_argument( "--same",
                        type = bool,
                     default=False)
    return parser.parse_args()
config = parse_args()
root = config.root
def recover(rt):
    rots = []
    trans = []
    for tt in range(rt.shape[0]):
        ret = rt[tt,:3]
        r = R.from_rotvec(ret)
        ret_R = r.as_dcm()
        ret_t = rt[tt, 3:]
        ret_t = ret_t.reshape(3,1)
        rots.append(ret_R)
        trans.append(ret_t)
    return (np.array(rots), np.array(trans))

def load_obj(obj_file):
    vertices = []

    triangles = []
    colors = []

    with open(obj_file) as infile:
        for line in infile.read().splitlines():
            if len(line) > 2 and line[:2] == "v ":
                ts = line.split()
                x = float(ts[1])
                y = float(ts[2])
                z = float(ts[3])
                r = float(ts[4])
                g = float(ts[5])
                b = float(ts[6])
                vertices.append([x,y,z])
                colors.append([r,g,b])
            elif len(line) > 2 and line[:2] == "f ":
                ts = line.split()
                fx = int(ts[1]) - 1
                fy = int(ts[2]) - 1
                fz = int(ts[3]) - 1
                triangles.append([fx,fy,fz])
    
    return (np.array(vertices), np.array(triangles).astype(np.int), np.array(colors))

def setup_renderer():    
    renderer = sr.SoftRenderer(camera_mode="look", viewing_scale=2/res, far=10000, perspective=False, image_size=res, camera_direction=[0,0,-1], camera_up=[0,1,0], light_intensity_ambient=1)
    renderer.transform.set_eyes([res/2, res/2, 6000])
    return renderer
def get_np_uint8_image(mesh, renderer):
    images = renderer.render_mesh(mesh)
    image = images[0]
    image = torch.flip(image, [1,2])
    image = image.detach().cpu().numpy().transpose((1,2,0))
    image = np.clip(image, 0, 1)
    image = (255*image).astype(np.uint8)
    return image




def demo_single_video(config, front_lmark_path = None ,  key_id = None, front_img_path=None, prnet_lmark_path=None, ref_lmark_path = None ):
    itvl = 1000.0/25.0 # 25fps
    overlay = False
    
    # extract the frontal facial landmarks for key frame'

    lmk3d_all = np.load(front_lmark_path)
    if config.same:
        print('fdjfkdjklfj===')
        lmk3d_target = lmk3d_all[key_id]
    else:
        lmk3d_target = np.load(ref_lmark_path)
    
        print(ref_lmark_path)
        print(lmk3d_target.shape)
    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load(prnet_lmark_path)
     
    print (lmk3d_target.shape, lmk3d_origin.shape,'+++++++')
    
    # load RTs for all frame
    rots, trans = recover(np.load( front_lmark_path[:-9] + "rt.npy"))

    # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj(front_lmark_path[:-9] +"original.obj") # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()
    ani_path =front_lmark_path[:-9] +'ani.mp4'
    if overlay:
        real_video = mmcv.VideoReader('/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00419/S8fiWqrZEew/00216_aligned.mp4')

    fig = plt.figure()
    ims = []
    temp_path = './tempp_00005'
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)
    face_mesh = sr.Mesh(vertices_org, triangles, colors, texture_type="vertex")
    image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
    
    # #####save rgba image as bgr in cv2
    # rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
    # # flipBoth = cv2.flip(rgb_frame, 1)
    # flipBoth = cv2.flip(rgb_frame, 0)
    # cv2.imwrite( temp_path +  "/" + id + "_original.png", flipBoth) 

    for i in range(rots.shape[0]):
        # if i == 510:
        #     break
        # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
        #save rgba image as bgr in cv2
        rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
        cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
    command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  ani_path
    os.system(command)
    print (command)


# target obj is the identity that you want, target front is the target mouth movement that you want, 
#original front and key load the original front face that paired with target indentity
# prnet is the pr path of the target identity
def demo_single_video_switch(target_obj_path = None, target_front_lmark_path = None, 
    target_rt_path = None,  original_front_lmark_path = None, original_key_id = None, pr_path = None, ani_save_path = None):
    itvl = 1000.0/25.0 # 25fps
    overlay = False
    
    # extract the frontal facial landmarks for key frame

    original_lmk3d_all = np.load(original_front_lmark_path)
    original_lmk3d_target = original_lmk3d_all[original_key_id]

    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load(pr_path)

    # load RTs for all frame
    rots, trans = recover(np.load( target_rt_path))

    # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, original_lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj(target_obj_path) # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()
    ani_path =ani_save_path
    
    fig = plt.figure()
    ims = []
    temp_path = './tempp_00005'
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)
    face_mesh = sr.Mesh(vertices_org, triangles, colors, texture_type="vertex")
    image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
    
    # #####save rgba image as bgr in cv2
    # rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
    # # flipBoth = cv2.flip(rgb_frame, 1)
    # flipBoth = cv2.flip(rgb_frame, 0)
    # cv2.imwrite( temp_path +  "/" + id + "_original.png", flipBoth) 

    for i in range(rots.shape[0]):
        # if i == 510:
        #     break
        # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
        #save rgba image as bgr in cv2
        rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
        cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
    command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  ani_path
    os.system(command)
    print (command)


def demo(id = 'lisa2'):
    key_id = 2433 # index of the frame used to do the 3D face reconstruction (key frame)
    itvl = 1000.0/25.0 # 25fps
    rt_id = '00025_aligned'
    overlay = False
    
    # extract the frontal facial landmarks for key frame'

    lmk3d_all = np.load("/home/cxu-serve/p1/common/demo/" + rt_id + "_front.npy")
    lmk3d_target = lmk3d_all[key_id]

    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load('/home/cxu-serve/p1/common/demo/' + id + '_crop_prnet.npy')
    lmk3d_origin[:,1] =  lmk3d_origin[:,1]

    # load RTs for all frame
    rots, trans = recover(np.load("/home/cxu-serve/p1/common/demo/"+ rt_id + "_rt.npy"))

    # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj("/home/cxu-serve/p1/common/demo/"+ id +"_crop_original.obj") # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()
    ani_path ='/home/cxu-serve/p1/common/demo/'+ rt_id + '__' + id +'_ani2.mp4'
    if overlay:
        real_video = mmcv.VideoReader('/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00419/S8fiWqrZEew/00216_aligned.mp4')

    fig = plt.figure()
    ims = []
    temp_path = './tempp_00005'
    # if os.path.exists(temp_path):
    #     shutil.rmtree(temp_path)
    # os.mkdir(temp_path)
    face_mesh = sr.Mesh(vertices_org, triangles, colors, texture_type="vertex")
    image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
    
    # #####save rgba image as bgr in cv2
    # rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
    # # flipBoth = cv2.flip(rgb_frame, 1)
    # flipBoth = cv2.flip(rgb_frame, 0)
    # cv2.imwrite( temp_path +  "/" + id + "_original.png", flipBoth) 

    for i in range(rots.shape[0]):
        if i == 510:
            break
        # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
        #save rgba image as bgr in cv2
        rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
        cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
    command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  ani_path
    os.system(command)
    print (command)

    #     if overlay:
    #         frame = mmcv.bgr2rgb(real_video[i]) # RGB, (224,224,3), np.uint8

    #     if not overlay:
    #         im = plt.imshow(image_render, animated=True)
    #     else:
    #         im = plt.imshow((frame[:,:,:3] * 0.5 + image_render[:,:,:3] * 0.5).astype(np.uint8), animated=True)
        
    #     ims.append([im])
    #     print("[{}/{}]".format(i+1, rots.shape[0])) 
    

    # ani = animation.ArtistAnimation(fig, ims, interval=itvl, blit=True, repeat_delay=1000)
    # if not overlay:
    #     ani.save('./_render.mp4')
    # else:
    #     ani.save('/home/cxu-serve/p1/common/demo/vincent2_ani.mp4')
    # plt.show()



def demo_obama():
    key_id = 405 # index of the frame used to do the 3D face reconstruction (key frame)
    itvl = 1000.0/25.0 # 25fps

    overlay = True
    
    # extract the frontal facial landmarks for key frame'

    lmk3d_all = np.load("/home/cxu-serve/p1/common/demo/00025_aligned_front.npy")
    lmk3d_target = lmk3d_all[key_id]

    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load('/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id04094/2sjuXzB2I1M/00025_prnet.npy')
    lmk3d_origin[:,1] =  lmk3d_origin[:,1]

    # load RTs for all frame
    rots, trans = recover(np.load("/home/cxu-serve/p1/common/demo/00025_aligned_rt.npy"))

    # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj("/home/cxu-serve/p1/common/Obama/video/3_3__original2.obj") # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()
    ani_path ='/home/cxu-serve/p1/common/demo/demo_00025_3_3__ani2.mp4'
    if overlay:
        real_video = mmcv.VideoReader('/home/cxu-serve/p1/common/Obama/video/3_3__crop2.mp4')

    fig = plt.figure()
    ims = []
    temp_path = './tempp_00005'
    # if os.path.exists(temp_path):
    #     shutil.rmtree(temp_path)
    # os.mkdir(temp_path)
    face_mesh = sr.Mesh(vertices_org, triangles, colors, texture_type="vertex")
    image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
    
    #####save rgba image as bgr in cv2
    # rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
    # # flipBoth = cv2.flip(rgb_frame, 1)
    # flipBoth = cv2.flip(rgb_frame, 0)
    # cv2.imwrite( temp_path +  "/demo_front.png", flipBoth) 

    for i in range(rots.shape[0]):
        # if i == 20:
        #     break
        # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
        #save rgba image as bgr in cv2
        rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
        cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
    
   
    command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  ani_path
    os.system(command)
    print (command)

def get_crema(batch = 0 ):
    # key_id = 58 #
    # model_id = "00025"
    root = '/home/cxu-serve/p1/common/CREMA'
    _file = open(os.path.join(root, 'pickle','train_lmark2img.pkl'), "rb")
    # _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    data = pickle.load(_file)
    _file.close()
    flage = False
    print (len(data))
    gg = data[int(0.2 * len(data)) * batch:int(0.2 * len(data)) * (batch + 1) ]
    for k, v_id in enumerate(gg):
        key_id = v_id[-1]

        video_path = os.path.join(root, 'VideoFlash', v_id[0][:-10] + '_crop.mp4'  )

        ani_path = os.path.join(root, 'VideoFlash', v_id[0][:-10] + '_ani.mp4'  )

        reference_img_path =  os.path.join(root, 'VideoFlash', v_id[0][:-10] + '_%05d.png'%key_id  )

        reference_prnet_lmark_path = os.path.join(root, 'VideoFlash', v_id[0][:-10] + '_prnet.npy')

        original_obj_path = os.path.join(root, 'VideoFlash', v_id[0][:-10] + '_original.obj') 

        rt_path = os.path.join(root,'VideoFlash', v_id[0][:-10] +'_rt.npy'  )

        lmark_path  =  os.path.join(root, 'VideoFlash', v_id[0][:-10] +'_front.npy'  )

        # if os.path.exists( ani_path):
        #     print ('=====')
        #     continue
        if  not os.path.exists(original_obj_path) or not os.path.exists(reference_prnet_lmark_path) or not os.path.exists(lmark_path) or not os.path.exists(rt_path):
            print (os.path.exists(original_obj_path) , os.path.exists(reference_prnet_lmark_path),  os.path.exists(lmark_path), os.path.exists(rt_path))
            print (original_obj_path)
            print ('++++')
            continue
        # try:
        # extract the frontal facial landmarks for key frame
        lmk3d_all = np.load(lmark_path)
        lmk3d_target = lmk3d_all[key_id]


        # load the 3D facial landmarks on the PRNet 3D reconstructed face
        lmk3d_origin = np.load(reference_prnet_lmark_path)
        # lmk3d_origin[:,1] = 32 +  lmk3d_origin[:,1]
        
        

        # load RTs
        rots, trans = recover(np.load(rt_path))

        # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
        lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
        p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
        pr = p_affine[:,:3] # 3x3
        pt = p_affine[:,3:] # 3x1

        # load the original 3D face mesh then transform it to align frontal face landmarks
        vertices_org, triangles, colors = load_obj(original_obj_path) # get unfrontalized vertices position
        vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

        # set up the renderer
        renderer = setup_renderer()
        # generate animation

        temp_path = './tempp_%05d'%batch

        # generate animation
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)
        # writer = imageio.get_writer('rotation.gif', mode='I')
        for i in range(rots.shape[0]):
                # get rendered frame
            vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
            face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
            image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
            
            #save rgba image as bgr in cv2
            rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
            cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
        command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  ani_path
        os.system(command)
        print (command)
        # break
        # except:
        #     print ('===++++')
        #     continue



def get_obama(batch = 0 ):
    # key_id = 58 #
    # model_id = "00025"
    root = '/home/cxu-serve/p1/common/Obama'
    _file = open(os.path.join(root, 'pickle','train_lmark2img.pkl'), "rb")
    # _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    data = pickle.load(_file)
    _file.close()
    flage = False
    print (len(data))
    gg = data[int(0.2 * len(data)) * batch:int(0.2 * len(data)) * (batch + 1) ]
    for k, v_id in enumerate(gg):
        key_id = v_id[-1]

        video_path = os.path.join(root, 'video', v_id[0][:-11] + '_crop2.mp4'  )

        ani_path = os.path.join(root, 'video', v_id[0][:-11] + '_ani2.mp4'  )

        reference_img_path =  os.path.join(root, 'video', v_id[0][:-11] + '_%05d_2.png'%key_id  )

        reference_prnet_lmark_path = os.path.join(root, 'video', v_id[0][:-11] + '_prnet2.npy')

        original_obj_path = os.path.join(root, 'video', v_id[0][:-11] + '_original2.obj') 

        rt_path = os.path.join(root,'video', v_id[0][:-11] +'_rt2.npy'  )

        lmark_path  =  os.path.join(root, 'video', v_id[0][:-11] +'_front2.npy'  )

        # if os.path.exists( ani_path):
        #     print ('=====')
        #     continue
        if  not os.path.exists(original_obj_path) or not os.path.exists(reference_prnet_lmark_path) or not os.path.exists(lmark_path) or not os.path.exists(rt_path):
            print (os.path.exists(original_obj_path) , os.path.exists(reference_prnet_lmark_path),  os.path.exists(lmark_path), os.path.exists(rt_path))
            print (original_obj_path)
            print ('++++')
            continue
        # try:
        # extract the frontal facial landmarks for key frame
        lmk3d_all = np.load(lmark_path)
        lmk3d_target = lmk3d_all[key_id]


        # load the 3D facial landmarks on the PRNet 3D reconstructed face
        lmk3d_origin = np.load(reference_prnet_lmark_path)
        # lmk3d_origin[:,1] = 32 +  lmk3d_origin[:,1]
        
        

        # load RTs
        rots, trans = recover(np.load(rt_path))

        # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
        lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
        p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
        pr = p_affine[:,:3] # 3x3
        pt = p_affine[:,3:] # 3x1

        # load the original 3D face mesh then transform it to align frontal face landmarks
        vertices_org, triangles, colors = load_obj(original_obj_path) # get unfrontalized vertices position
        vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

        # set up the renderer
        renderer = setup_renderer()
        # generate animation

        temp_path = './tempp_%05d'%batch

        # generate animation
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)
        # writer = imageio.get_writer('rotation.gif', mode='I')
        for i in range(rots.shape[0]):
                # get rendered frame
            vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
            face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
            image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
            
            #save rgba image as bgr in cv2
            rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
            cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
        command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  ani_path
        os.system(command)
        print (command)
        # break
        # except:
        #     print ('===++++')
        #     continue
def get_lrw(batch = 0 ):
    # key_id = 58 #
    # model_id = "00025"
    root = '/home/cxu-serve/p1/common/lrw'
    _file = open(os.path.join(root, 'pickle','test2_lmark2img.pkl'), "rb")
    data = pickle.load(_file)
    _file.close()
    flage = False
    print (len(data))
    gg = data[int(0.2 * len(data)) * batch:int(0.2 * len(data)) * (batch + 1) ]
    for k, v_id in enumerate(gg):
        print (v_id)
        key_id = v_id[-1]
        ani_path =v_id[0] + '_ani.mp4'  
        reference_prnet_lmark_path = v_id[0] + '_prnet.npy'

        original_obj_path = v_id[0] + '_original.obj'

        rt_path = v_id[0]  +'_rt.npy'  

        lmark_path  =  v_id[0]  +'_front.npy'  

        if os.path.exists( ani_path):
            print ('=====')
            continue
        if  not os.path.exists(original_obj_path) or not os.path.exists(reference_prnet_lmark_path) or not os.path.exists(lmark_path) or not os.path.exists(rt_path):
            print (os.path.exists(original_obj_path) , os.path.exists(reference_prnet_lmark_path),  os.path.exists(lmark_path), os.path.exists(rt_path))
            print (original_obj_path)
            print ('++++')
            continue
        # try:
        # extract the frontal facial landmarks for key frame
        lmk3d_all = np.load(lmark_path)
        lmk3d_target = lmk3d_all[key_id]


        # load the 3D facial landmarks on the PRNet 3D reconstructed face
        lmk3d_origin = np.load(reference_prnet_lmark_path)
        # lmk3d_origin[:,1] = 32 +  lmk3d_origin[:,1]
        
        

        # load RTs
        rots, trans = recover(np.load(rt_path))

        # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
        lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
        p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
        pr = p_affine[:,:3] # 3x3
        pt = p_affine[:,3:] # 3x1

        # load the original 3D face mesh then transform it to align frontal face landmarks
        vertices_org, triangles, colors = load_obj(original_obj_path) # get unfrontalized vertices position
        vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

        # set up the renderer
        renderer = setup_renderer()
        # generate animation

        temp_path = './tempp_%05d'%batch

        # generate animation
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)
        # writer = imageio.get_writer('rotation.gif', mode='I')
        for i in range(rots.shape[0]):
                # get rendered frame
            vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
            face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
            image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
            
            #save rgba image as bgr in cv2
            rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
            cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
        command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  ani_path
        os.system(command)
        print (command)
        # break
        # except:
        #     print ('===++++')
        #     continue



def get_lrs(batch = 0 ):
    # key_id = 58 #
    # model_id = "00025"
    root = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4'
    _file = open(os.path.join(root, 'pickle','test2_lmark2img.pkl'), "rb")
    # _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    data = pickle.load(_file)
    _file.close()
    flage = False
    print (len(data))
    for k, v_id in enumerate(data):
        key_id = v_id[-1]

        video_path = os.path.join(root, 'test', v_id[0] , v_id[1][:5] + '_crop.mp4'  )

        ani_path = os.path.join(root, 'test', v_id[0] , v_id[1][:5] + '_ani.mp4'  )

        reference_img_path =  os.path.join(root, 'test', v_id[0] , v_id[1][:5] + '_%05d.png'%key_id  )

        reference_prnet_lmark_path = os.path.join(root, 'test', v_id[0] , v_id[1][:5] + '_prnet.npy')

        original_obj_path = os.path.join(root, 'test', v_id[0] , v_id[1][:5] + '_original.obj') 

        rt_path = os.path.join(root, 'test', v_id[0] , v_id[1][:5] +'_rt.npy'  )

        lmark_path  =  os.path.join(root, 'test', v_id[0] , v_id[1][:5] +'_front.npy'  )

        if os.path.exists( ani_path):
            print (ani_path)
            print ('=====')
            continue
        if  not os.path.exists(original_obj_path) or not os.path.exists(reference_prnet_lmark_path) or not os.path.exists(lmark_path) or not os.path.exists(rt_path):
            print (os.path.exists(original_obj_path) , os.path.exists(reference_prnet_lmark_path),  os.path.exists(lmark_path), os.path.exists(rt_path))
            print (original_obj_path)
            print ('++++')
            continue
        # try:
        # extract the frontal facial landmarks for key frame
        lmk3d_all = np.load(lmark_path)
        lmk3d_target = lmk3d_all[key_id]


        # load the 3D facial landmarks on the PRNet 3D reconstructed face
        lmk3d_origin = np.load(reference_prnet_lmark_path)
        # lmk3d_origin[:,1] = lmk3d_origin[:,1] + 32
        # load RTs
        rots, trans = recover(np.load(rt_path))

        # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
        lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
        p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
        pr = p_affine[:,:3] # 3x3
        pt = p_affine[:,3:] # 3x1

        # load the original 3D face mesh then transform it to align frontal face landmarks
        vertices_org, triangles, colors = load_obj(original_obj_path) # get unfrontalized vertices position
        vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

        # set up the renderer
        renderer = setup_renderer()
        # generate animation

        temp_path = './tempp_%05d'%batch

        # generate animation
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)
        # writer = imageio.get_writer('rotation.gif', mode='I')
        for i in range(rots.shape[0]):
                # get rendered frame
            vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
            face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
            image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
            
            #save rgba image as bgr in cv2
            rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
            cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
        command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  ani_path
        os.system(command)
        print (command)
        # break
        # except:
        #     print ('===++++')
        #     continue

def vis_single(video_path, key_id, save_name):
    overlay = True
    #key_id = 79 #
    #video_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/test_video/id04276/k0zLls_oen0/00341.mp4'
    reference_img_path = video_path[:-4] + '_%05d.png'%key_id
    reference_prnet_lmark_path = video_path[:-4] +'_prnet.npy'

    original_obj_path = video_path[:-4] + '_original.obj'

    rt_path  = video_path[:-4] + '_sRT.npy'

    lmark_path  = video_path[:-4] +'_front.npy'



    # extract the frontal facial landmarks for key frame
    lmk3d_all = np.load(lmark_path)
    lmk3d_target = lmk3d_all[key_id]


    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load(reference_prnet_lmark_path)
    # lmk3d_origin[:,1] = res - lmk3d_origin[:,1]


    # load RTs
    rots, trans = recover(np.load(rt_path))

     # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj(original_obj_path) # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()
    # generate animation

    if os.path.exists('./tempo1'):
        shutil.rmtree('./tempo1')
    os.mkdir('./tempo1')
    if overlay:
        real_video = mmcv.VideoReader(video_path)

    for i in range(rots.shape[0]):
        t = time.time()
        

        # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
        print (image_render.shape)
        print (image_render.max())
        print (image_render.min())
        #save rgba image as bgr in cv2
        rgb_frame =  (image_render ).astype(int)[:,:,:-1][...,::-1]
        overla_frame = (0.5* rgb_frame + 0.5 * real_video[i]).astype(int)
        cv2.imwrite("./tempo1/%05d.png"%i, overla_frame)


        print (time.time() - t)
        # writer.append_data((255*warped_image).astype(np.uint8))

        print("[{}/{}]".format(i+1, rots.shape[0]))    
        # if i == 5:
        #     breakT
    t = time.time()
    ani_mp4_file_name = save_name  # './fuck.mp4'
    command = 'ffmpeg -framerate 25 -i ./tempo1/%5d.png  -c:v libx264 -y -vf format=yuv420p ' + ani_mp4_file_name 
    os.system(command)
    print (time.time() - t)
    
import random    
def gg():
    _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    data = pickle._Unpickler(_file)
    data.encoding = 'latin1'

    data = data.load()
    _file.close()
    print (len(data))
    # random.shuffle(data)
    for k, item in enumerate(data):



        key_id = item[-1]
        
        video_path = os.path.join(root, 'unzip', item[0] + '_ani.mp4')
        # save_name = './tempo2/' + item[0].replace('/', '_') + '.mp4' 
        if k % 10 ==0 and k > 1800 and k < 1900:
            print (video_path)
        # vis_single(video_path, key_id, save_name)
        # if k == 10:
        #     break
# demo('vincent2')



    parser.add_argument("--front_img_path",
                        type=str,
                        default='')
    parser.add_argument("--front_lmark_path",
                        type=str,
                        default='')
    
    parser.add_argument("--prnet_lmark_path",
                        type=str,
                        default='')

    
    
def main():
    config = parse_args()
    front_img_path = config.front_img_path
    front_lmark_path = config.front_lmark_path
    prnet_lmark_path = config.prnet_lmark_path
    same = config.same
    if same :
        front_frame_id =  int(front_img_path[-9 : -4])
        demo_single_video(config, front_lmark_path = front_lmark_path , key_id =front_frame_id , prnet_lmark_path = prnet_lmark_path)

    else:
        front_frame_id = None
        ref_lmark_path = config.ref_lmark_path 
        demo_single_video(config, front_lmark_path = front_lmark_path  , front_img_path = front_img_path, prnet_lmark_path = prnet_lmark_path, ref_lmark_path = ref_lmark_path)

 

main()

# parser.add_argument("--front_lmark_path", '/u/lchen63/github/Talking-head-Generation-with-Rhythmic-Head-Motion/test_data/ouyang__front.npy'
#                         type=str,
#                         default='')
#     parser.add_argument("--front_frame_id",
#                         type=int,
#                         default=1)


# demo_single_video_switch(target_obj_path = '/home/cxu-serve/p1/common/demo/oppo_demo/ouyang__original.obj', 
#                         target_front_lmark_path = '/home/cxu-serve/p1/common/demo/oppo_demo/957__front.npy',  
#                         target_rt_path = '/home/cxu-serve/p1/common/demo/oppo_demo/957__rt.npy',  
#                         original_front_lmark_path = '/home/cxu-serve/p1/common/demo/oppo_demo/ouyang__front.npy', 
#                         original_key_id = 11174, 
#                         pr_path = "/home/cxu-serve/p1/common/demo/oppo_demo/ouyang__prnet.npy", 
#                         ani_save_path = '/home/cxu-serve/p1/common/demo/oppo_demo/ouyang__957.mp4')
# demo_obama()
# gg()
# demo('vincent2')
# get_crema(config.b)
# get_lrs(config.b)
# get_lrw(config.b)
# print ('cool')
# get_obama(config.b)
