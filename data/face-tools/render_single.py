import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import os
import cv2
import mmcv
import torch
import soft_renderer as sr

res = 256
overlay = False

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

def generatert():
  
    itvl = 1000.0/25.0 # 25fps
    
    # extract the frontal facial landmarks for key frame
    lmk3d_target = np.load("/home/cxu-serve/p1/common/demo/lisa2_original.npy")

    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load("/home/cxu-serve/p1/common/demo/lisa2_crop_prnet.npy")


    # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj("/home/cxu-serve/p1/common/demo/lisa2_crop_original.obj") # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    n_frames = 200 # number of RT you want
    rots = np.zeros((n_frames, 3, 3))
    trans = np.zeros((n_frames, 3, 1))
    t = vertices_origin_affine.mean(axis=0).reshape(3,1)
    for i in range(n_frames):
        rots[i,:,:] = R.from_euler("xyz", [0, np.pi * 0.5 - np.pi / n_frames * i, 0]).as_dcm()
        trans[i,:,:] = t - rots[i,:,:] @ t

    rt = rots.copy()
    for j in range(200):
        ret_R = rt[j]
        r = Rotation.from_dcm(ret_R)
        vec = r.as_rotvec()             
        RTs[j,:3] = vec
        RTs[j,3:] =  np.squeeze(np.asarray(ret_t))           

    # save the RT now
    np.save("/home/cxu-serve/p1/common/demo/lisa2_rt.npy", rots)
    ani_path =  "/home/cxu-serve/p1/common/demo/lisa2_crop_ani.mp4"
    # lele: XXXXXX

    # set up the renderer
    renderer = setup_renderer()

    fig = plt.figure()
    ims = []
    s
    for i in range(rots.shape[0]):
        # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        # vertices = vertices_origin_affine
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8
        # im = plt.imshow(image_render, animated=True)
        # ims.append([im])
        # print("[{}/{}]".format(i+1, rots.shape[0])) 
        #save rgba image as bgr in cv2
        rgb_frame =  (image_render).astype(int)[:,:,:-1][...,::-1]
        cv2.imwrite( temp_path +  "/%05d.png"%i, rgb_frame)  
    command = 'ffmpeg -framerate 25 -i '  + temp_path + '/%5d.png  -c:v libx264 -y -vf format=yuv420p ' +  ani_path
    os.system(command)

generatert()