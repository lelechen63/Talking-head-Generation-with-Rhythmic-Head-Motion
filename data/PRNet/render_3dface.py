import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

# import cv2
import mmcv
import torch
import soft_renderer as sr

res = 224
overlay = True

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

if __name__ == "__main__":
    key_id = 79 # index of the frame used to do the 3D face reconstruction (key frame)
    model_id = "00341"
    itvl = 1000.0/25.0 # 25fps
    
    # extract the frontal facial landmarks for key frame
    lmk3d_all = np.load("{}/{}_front.npy".format(model_id, model_id))
    lmk3d_target = lmk3d_all[key_id]

    # load the 3D facial landmarks on the PRNet 3D reconstructed face
    lmk3d_origin = np.load("{}/{}_prnet.npy".format(model_id, model_id))
    # lmk3d_origin[:,1] = res - lmk3d_origin[:,1]

    # load RTs for all frame
    rots, trans = recover(np.load("{}/{}_sRT.npy".format(model_id, model_id)))

    # calculate the affine transformation between PRNet 3D face and the frotal face landmarks
    lmk3d_origin_homo = np.hstack((lmk3d_origin, np.ones([lmk3d_origin.shape[0],1]))) # 68x4
    p_affine = np.linalg.lstsq(lmk3d_origin_homo, lmk3d_target, rcond=1)[0].T # Affine matrix. 3 x 4
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1

    # load the original 3D face mesh then transform it to align frontal face landmarks
    vertices_org, triangles, colors = load_obj("{}/{}_original.obj".format(model_id, model_id)) # get unfrontalized vertices position
    vertices_origin_affine = (pr @ (vertices_org.T) + pt).T # aligned vertices

    # set up the renderer
    renderer = setup_renderer()

    if overlay:
        real_video = mmcv.VideoReader("{}/{}.mp4".format(model_id, model_id))

    fig = plt.figure()
    ims = []

    for i in range(rots.shape[0]):
        # get rendered frame
        vertices = (rots[i].T @ (vertices_origin_affine.T - trans[i])).T
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type="vertex")
        image_render = get_np_uint8_image(face_mesh, renderer) # RGBA, (224,224,3), np.uint8

        if overlay:
            frame = mmcv.bgr2rgb(real_video[i]) # RGB, (224,224,3), np.uint8

        if not overlay:
            im = plt.imshow(image_render, animated=True)
        else:
            im = plt.imshow((frame[:,:,:3] * 0.5 + image_render[:,:,:3] * 0.5).astype(np.uint8), animated=True)
        
        ims.append([im])
        print("[{}/{}]".format(i+1, rots.shape[0])) 
    

    ani = animation.ArtistAnimation(fig, ims, interval=itvl, blit=True, repeat_delay=1000)
    if not overlay:
        ani.save('{}/{}_render.mp4'.format(model_id, model_id))
    else:
        ani.save('{}/{}_overlay.mp4'.format(model_id, model_id))
    plt.show()



"""

    plt.subplot(1,2,1)
    lmk3d = (rots[key_id].T @ (lmk3d_target.T - trans[key_id])).T
    preds = lmk3d
    plt.imshow(image)
    plt.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)

    plt.subplot(1,2,2)
    lmk3d_origin_affine = (pr @ (lmk3d_origin.T) + pt).T
    lmk3d = (rots[key_id].T @ (lmk3d_origin_affine.T - trans[key_id])).T
    preds = lmk3d
    plt.imshow(image)
    plt.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    plt.show()

"""
