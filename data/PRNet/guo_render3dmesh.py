import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

# import cv2
import mmcv
import torch
import soft_renderer as sr
import tqdm

res = 224
overlay = False


def recover(rt):
    rots = []
    trans = []
    for tt in range(rt.shape[0]):
        ret = rt[tt, :3]
        r = R.from_rotvec(ret)
        ret_R = r.as_dcm()
        ret_t = rt[tt, 3:]
        ret_t = ret_t.reshape(3, 1)
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
                vertices.append([x, y, z])
                colors.append([r, g, b])
            elif len(line) > 2 and line[:2] == "f ":
                ts = line.split()
                fx = int(ts[1]) - 1
                fy = int(ts[2]) - 1
                fz = int(ts[3]) - 1
                triangles.append([fx, fy, fz])

    return (np.array(vertices), np.array(triangles).astype(np.int), np.array(colors))


def setup_renderer():
    # renderer = sr.SoftRenderer(camera_mode="look", viewing_scale=2 / res, far=10000, perspective=False, image_size=res,
    #                            camera_direction=[0, 0, -1], camera_up=[0, 1, 0], light_intensity_ambient=1)
    # renderer.transform.set_eyes([res / 2, res / 2, 6000])

    renderer = sr.SoftRenderer(camera_mode="look_at")
    return renderer


def get_np_uint8_image(mesh, renderer):
    images = renderer.render_mesh(mesh)
    image = images[0]
    image = torch.flip(image, [1, 2])
    image = image.detach().cpu().numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    image = (255 * image).astype(np.uint8)
    return image

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

if __name__ == "__main__":
    # key_id = 79  # index of the frame used to do the 3D face reconstruction (key frame)
    # model_id = "00341"
    itvl = 1000.0 / 25.0  # 25fps

    key_id = 0
    source_path = '/home/cxu-serve/p1/common/RyersonAudioVisual/Video_Speech_Actor/Actor_01/'
    audio_path = '/home/cxu-serve/p1/common/RyersonAudioVisual/Audio_Speech_Actors_01-24/Actor_01/'
    expression_id = "01"
    video_path = source_path + '01-01-%s-01-01-01-01.mp4' % expression_id
    mesh_path = source_path + "3dmesh"
    mesh_id = "01/01-01-%s-01-01-01-01" % expression_id

    renderer = setup_renderer()
    if overlay:
        real_video = mmcv.VideoReader(video_path)

    fig = plt.figure()
    ims = []

    num_frames = count_frames(video_path)
    for i in range(num_frames):
        vertices, triangles, colors = load_obj("{}/{}_%05d_original.obj".format(mesh_path, mesh_id) % i)
        face_mesh = sr.Mesh(vertices, triangles, colors, texture_type='vertex')
        image_render = get_np_uint8_image(face_mesh, renderer)

        if overlay:
            frame = mmcv.bgr2rgb(real_video[i])  # RGB, (224,224,3), np.uint8

        if not overlay:
            im = plt.imshow(image_render, animated=True)
        else:
            im = plt.imshow((frame[:, :, :3] * 0.5 + image_render[:, :, :3] * 0.5).astype(np.uint8), animated=True)

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=itvl, blit=True, repeat_delay=1000)
    if not overlay:
        ani.save('{}/{}_render.mp4'.format(mesh_path, mesh_id))
    else:
        ani.save('{}/{}_overlay.mp4'.format(mesh_path, mesh_id))
    plt.savefig("{}/{}_fig".format(mesh_path, mesh_id))



