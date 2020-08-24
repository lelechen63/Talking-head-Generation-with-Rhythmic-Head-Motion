import os
import argparse
import shutil
from tqdm import tqdm
import glob, os
import face_alignment
import numpy as np
import cv2
from face_tracker import _crop_video
from utils import face_utils, util
from scipy.spatial.transform import Rotation 
from scipy.io import wavfile
import torch

# from .dp2model import load_model
# from .dp2dataloader import SpectrogramParser

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', "--batch_id",
                     type=int,
                     default=1)
    
    return parser.parse_args()
config = parse_args()


def read_videos( video_path):
    cap = cv2.VideoCapture(video_path)
    real_video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            real_video.append(frame)
        else:
            break

    return real_video

def landmark_extractor( video_path = None, path = None):
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0')


	if video_path != None:
		print ('------++++++++++-')

		tmp = video_path.split('/')
		path = os.path.join( '/',  *tmp[:-1] )
		p_id = tmp[-1]
		original_video_path =video_path
		lmark_path = os.path.join(path,   p_id[:-4] + '__original.npy')            
		print (original_video_path)
		cropped_video_path = os.path.join(path,   p_id[:-4] + '_crop.mp4')

		# try:
			# try:
			# 	_crop_video(original_video_path, config.batch_id,  1)
			# except:

			# 	print('some error when crop images.')
		command = 'ffmpeg -framerate 25  -i ./temp%05d'%config.batch_id + '/%05d.png  -vcodec libx264  -vf format=yuv420p -y ' +  cropped_video_path
		os.system(command)
		cap = cv2.VideoCapture(cropped_video_path)
		lmark = []
		print ('-------')
		while(cap.isOpened()):
			# counter += 1 
			# if counter == 5:
			#     break
			ret, frame = cap.read()
			if ret == True:
				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB )
				preds = fa.get_landmarks(frame)[0]
				lmark.append(preds)
			else:
				break
		lmark = np.asarray(lmark)
		np.save(lmark_path, lmark)

		print 
		# except:
		# 	print (cropped_video_path)
	else:
		train_list = sorted(os.listdir(path))
		batch_length =   len(train_list)
		for i in tqdm(range(batch_length)):
		    p_id = train_list[i]
		    if 'crop' in p_id or p_id[-3:] == 'npy':
		        continue

		    original_video_path = os.path.join( path,  p_id)
		    lmark_path = os.path.join(path,   p_id[:-4] + '__original.npy')            
		    print (original_video_path)
		    cropped_video_path = os.path.join(path,   p_id[:-4] + '_crop.mp4')
		    
		    # if os.path.exists(lmark_path):
		        # continue
		        
		    try:
		        # _crop_video(original_video_path, config.batch_id,  1)
		        
		        command = 'ffmpeg -framerate 25  -i ./temp%05d'%config.batch_id + '/%05d.png  -vcodec libx264  -vf format=yuv420p -y ' +  cropped_video_path
		        os.system(command)
		        cap = cv2.VideoCapture(cropped_video_path)
		        lmark = []
		        while(cap.isOpened()):
		            # counter += 1 
		            # if counter == 5:
		            #     break
		            ret, frame = cap.read()
		            if ret == True:
		                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB )

		                preds = fa.get_landmarks(frame)[0]
		                lmark.append(preds)
		            else:
		                break
		                
		        lmark = np.asarray(lmark)
		        np.save(lmark_path, lmark)
		    except:
		        print (cropped_video_path)

		        continue

def RT_compute(video_path = None, path  = None):  #video  path should be the original video path
	consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]  
	source = np.zeros((len(consider_key),3))
	ff = np.load('../basics/standard.npy')
	for m in range(len(consider_key)):
		source[m] = ff[consider_key[m]]  
	source = np.mat(source)
	if video_path != None:
		lmark_path = video_path[:-4] + '__original.npy'
		rt_path = video_path[:-4] +'__rt.npy'
		front_path = video_path[:-4]+'__front.npy'
		# normed_path  = os.path.join( person_path,vid[:-12] +'normed.npy')
		if os.path.exists(front_path):
		    print ('No front path ')
		if not os.path.exists(lmark_path):
		    print ('No landmark path ')
		lmark = np.load(lmark_path)
		############################################## smooth the landmark
		length = lmark.shape[0] 
		lmark_part = np.zeros((length,len(consider_key),3))
		RTs =  np.zeros((length,6))
		frontlized =  np.zeros((length,68,3))
		for j in range(length ):
			for m in range(len(consider_key)):
				lmark_part[:,m] = lmark[:,consider_key[m]] 

			target = np.mat(lmark_part[j])
			ret_R, ret_t = face_utils.rigid_transform_3D( target, source)
			source_lmark  = np.mat(lmark[j])
			A2 = ret_R*source_lmark.T
			A2+= np.tile(ret_t, (1, 68))
			A2 = A2.T
			frontlized[j] = A2
			r = Rotation.from_dcm(ret_R)
			vec = r.as_rotvec()             
			RTs[j,:3] = vec
			RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
		np.save(rt_path, RTs)
		np.save(front_path, frontlized)

	else:

	    train_list = sorted(os.listdir(path))
	    batch_length = int( len(train_list))
	   
	    for i in tqdm(range(batch_length)):
	        p_id = train_list[i]
	        if p_id[-3:] !=  'mp4':
	            continue
	            
	        # if 'crop' in p_id:
	        #     continue
	        lmark_path = os.path.join( path,   p_id[:-4] + '__original.npy')  
	        
	        rt_path = os.path.join( path , p_id[:-4] +'__rt.npy')
	        front_path = os.path.join(  path, p_id[:-4] +'__front.npy')
	        # normed_path  = os.path.join( person_path,vid[:-12] +'normed.npy')
	        if os.path.exists(front_path):
	            continue
	        if not os.path.exists(lmark_path):
	            continue
	        lmark = np.load(lmark_path)
	        ############################################## smooth the landmark
	      
	        length = lmark.shape[0] 
	        lmark_part = np.zeros((length,len(consider_key),3))
	        RTs =  np.zeros((length,6))
	        frontlized =  np.zeros((length,68,3))
	        for j in range(length ):
	            for m in range(len(consider_key)):
	                lmark_part[:,m] = lmark[:,consider_key[m]] 

	            target = np.mat(lmark_part[j])
	            ret_R, ret_t = face_utils.rigid_transform_3D( target, source)

	            source_lmark  = np.mat(lmark[j])

	            A2 = ret_R*source_lmark.T
	            A2+= np.tile(ret_t, (1, 68))
	            A2 = A2.T
	            frontlized[j] = A2
	            r = Rotation.from_dcm(ret_R)
	            vec = r.as_rotvec()             
	            RTs[j,:3] = vec
	            RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
	        np.save(rt_path, RTs)
	        np.save(front_path, frontlized)
	    print (front_path)
            # break
        # break
import torch
import random
from sklearn.decomposition import PCA
from utils import face_utils

def openrate(lmark1):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    for k in range(3):
        open_rate1.append(np.absolute(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2]))
        
    open_rate1 = np.asarray(open_rate1)
    return open_rate1.mean()
import pickle as pkl
def pca_lmark_grid():
    root_path  ='/home/cxu-serve/p1/common/grid'
    _file = open(os.path.join(root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
    datalist = pkl.load(_file)
    _file.close()
    batch_length = int( len(datalist))
    landmarks = []
    k = 20
    norm_lmark = np.load('../basics/s1_pgbk6n_01.npy')
   
    for index in tqdm(range(batch_length)):
        # if index == 10:
        #     break
        lmark_path = os.path.join(root_path ,  'align' , datalist[index][0] , datalist[index][1] + '_front.npy') 
        lmark = np.load(lmark_path)[:,:,:2]
        # if lmark.shape[0]< 74:
        #     continue

        openrates = []
        for  i in range(lmark.shape[0]):
            openrates.append(openrate(lmark[i]))
        openrates = np.asarray(openrates)
        min_index = np.argmin(openrates)
        diff =  lmark[min_index] - norm_lmark
        np.save(lmark_path[:-10] +'_%05d_diff.npy'%(min_index) , diff)
        datalist[index].append(min_index) 
    #     lmark = lmark - diff
    #     if datalist[index][2] == True: 
    #         indexs = random.sample(range(0,10), 6)
    #         for i in indexs:
    #             landmarks.append(lmark[i])
    #     if datalist[index][3] == True: 
    #         indexs = random.sample(range(65,74), 6)
    #         for i in indexs:
    #             landmarks.append(lmark[i])

    #     indexs = random.sample(range(11,65), 10)
    #     for i in indexs:
    #         landmarks.append(lmark[i])
       
    # landmarks = np.stack(landmarks)
    # print (landmarks.shape)
    # landmarks = landmarks.reshape(landmarks.shape[0], 136)
    # pca = PCA(n_components=20)
    # pca.fit(landmarks)
    
    # np.save('../basics/mean_grid_front.npy', pca.mean_)
    # np.save('../basics/U_grid_front.npy',  pca.components_)
    with open(os.path.join(root_path, 'pickle','test_audio2lmark_grid.pkl'), 'wb') as handle:
        pkl.dump(datalist, handle, protocol=pkl.HIGHEST_PROTOCOL)




def get_front_video(video_path): # video path should be the video path of cropped video.
   		

    v_frames = read_videos(video_path)
    # tmp = video_path.split('/')
    rt_path = video_path[:-9] + '__rt.npy'
    rt = np.load(rt_path)
    lmark_length = rt.shape[0]
    find_rt = []
    for t in range(0, lmark_length):
        find_rt.append(sum(np.absolute(rt[t,:3])))
    find_rt = np.asarray(find_rt)

    min_index = np.argmin(find_rt)
    
    img_path = video_path[:-9] + '__%05d.png'%min_index 

    print (img_path)
    cv2.imwrite(img_path, v_frames[min_index])
    

def swith_identity_obama(obamaid = '00025_aligned'):

    # file:///home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id04094/2sjuXzB2I1M/00025.mp4
        # src_lmark_path = '/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00817/HUmfsvegMRo/00175_aligned_front.npy' 
        src_lmark_path = '/home/cxu-serve/p1/common/Obama/video/3_3__front2.npy' 
        # tar_lmark_path =  '/home/cxu-serve/p1/common/demo/'  +  id + '_original_front.npy'
        tar_lmark_path = '/home/cxu-serve/p1/common/demo/00025__lisa2__rotated.npy'

        tar_front_lmark_path = '/home/cxu-serve/p1/common/demo/00025__lisa2__front.npy'
        srt_rt_path = '/home/cxu-serve/p1/common/demo/00025_aligned_rt.npy'
        print (srt_rt_path)
        rt = np.load(srt_rt_path)
        # rt = np.vstack([rt,rt[::-1,:],rt,rt[::-1,:],rt,rt[::-1,:],rt,rt[::-1,:],rt,rt[::-1,:],rt,rt[::-1,:]])

        # np.save('/home/cxu-serve/p1/common/demo/3_3__rt2.npy', rt)

        print (rt.shape)
        
        src_lmark = np.load(src_lmark_path)




        
        tar_lmark = np.load(tar_lmark_path)

        # tar_lmark = np.vstack([tar_lmark,tar_lmark[::-1,:,:],tar_lmark,tar_lmark[::-1,:,:],tar_lmark,tar_lmark[::-1,:,:],tar_lmark,tar_lmark[::-1,:,:],tar_lmark,tar_lmark[::-1,:,:],tar_lmark,tar_lmark[::-1,:,:]])

        # np.save('/home/cxu-serve/p1/common/demo/3_3__original2.npy', tar_lmark)
        lmark_length = min(src_lmark.shape[0], tar_lmark.shape[0] )        
        find_rt = []
        for t in range(0, lmark_length):
            find_rt.append(sum(np.absolute(rt[t,:3])))
        find_rt = np.asarray(find_rt)


        tar_front_lmark = np.load(tar_front_lmark_path)

        # tar_front_lmark = np.vstack([tar_front_lmark,tar_front_lmark[::-1,:,:],tar_front_lmark,tar_front_lmark[::-1,:,:],tar_front_lmark,tar_front_lmark[::-1,:,:],tar_front_lmark,tar_front_lmark[::-1,:,:],tar_front_lmark,tar_front_lmark[::-1,:,:],tar_front_lmark,tar_front_lmark[::-1,:,:]])
        # np.save('/home/cxu-serve/p1/common/demo/3_3__front2.npy', tar_front_lmark)
        min_indexs =  np.argsort(find_rt)[:100]

        for indx in min_indexs:
            if  abs(openrate(tar_lmark[indx])) < 2:
                min_index = indx
                tempolate_openrate = openrate(tar_front_lmark[indx])
                break

        
        current_tempolate =tar_front_lmark[min_index].copy()



        for  i in range(lmark_length):
            if abs(openrate(src_lmark[i])- tempolate_openrate) < 0.1:
                src_id = i
                break

        print (min_index, src_id)

        diff = current_tempolate[:,:2] -  src_lmark[src_id,:,:2]
              

        denormed_lamrk = tar_front_lmark[:lmark_length].copy()
        
        denormed_lamrk[:,48:,:2] = src_lmark[:,48:,:2] # + diff[48:]

        rotated = np.zeros((lmark_length, 68 , 3))
        for i in range(denormed_lamrk.shape[0]):
            rotated[i] = util.reverse_rt(denormed_lamrk[i], rt[i])


        np.save('/home/cxu-serve/p1/common/demo/demo_' + obamaid + '__rotated.npy', rotated)
        np.save('/home/cxu-serve/p1/common/demo/demo_' + obamaid + '__front.npy', denormed_lamrk)



def swith_identity(id = 'lisa2'):
        src_lmark_path = '/home/cxu-serve/p1/common/demo/00025_aligned_front.npy' 

        tar_lmark_path =  '/home/cxu-serve/p1/common/demo/'  +  id + '_original_front.npy'

        srt_rt_path = src_lmark_path.replace('front', 'rt')
        rt = np.load(srt_rt_path)
        lmark_length = rt.shape[0]


        src_lmark = np.load(src_lmark_path)

        tar_lmark = np.load(tar_lmark_path)[0]


        openrates = []
        for  i in range(src_lmark.shape[0]):
            openrates.append(openrate(src_lmark[i]))
        openrates = np.asarray(openrates)
        min_index = np.argmin(openrates)

        print (min_index)

        current_tempolate =src_lmark[min_index].copy()

        diff = current_tempolate[:,:2] -  tar_lmark[:,:2]

        denormed_lamrk = src_lmark.copy()
        denormed_lamrk[:,:48,:2] = src_lmark[:,:48,:2]  - diff[:48]

        rotated = np.zeros((denormed_lamrk.shape[0], 68 , 3))
        for i in range(denormed_lamrk.shape[0]):
            rotated[i] = util.reverse_rt(denormed_lamrk[i], rt[i])

        np.save('/home/cxu-serve/p1/common/demo/00025__' + id + '__rotated.npy', rotated)
        np.save('/home/cxu-serve/p1/common/demo/00025__' + id + '__front.npy', denormed_lamrk)


    #     find_rt = []
    #     for t in range(0, lmark_length):
    #         find_rt.append(sum(np.absolute(rt[t,:3])))
    #     find_rt = np.asarray(find_rt)

    #     min_index = np.argmin(find_rt)
        
    #     img_path =  os.path.join(root,  'video', v_id[0][:-11] + '_%05d_2.png'%min_index  )
    #     cv2.imwrite(img_path, v_frames[min_index])
    #     data[index].append(min_index)
    # with open(os.path.join( root, 'pickle','train_lmark2img.pkl'), 'wb') as handle:
    #     pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

def diff():
    root_path  = '/home/cxu-serve/p1/common/CREMA'
    _file = open(os.path.join(root_path, 'pickle','train_lmark2img.pkl'), "rb")
    datalist = pkl.load(_file)
    _file.close()
    batch_length = int( len(datalist))
    landmarks = []
    k = 20
    norm_lmark = np.load('../basics/s1_pgbk6n_01.npy')[:,:2]
   
    for index in tqdm(range(batch_length)):
        lmark_path = os.path.join(root_path,  'VideoFlash', datalist[index][0][:-10] +'_front.npy'  )
        lmark = np.load(lmark_path)[:,:,:2]


        openrates = []
        for  i in range(lmark.shape[0]):
            openrates.append(openrate(lmark[i]))
        openrates = np.asarray(openrates)
        min_index = np.argmin(openrates)
        diff =  lmark[min_index] - norm_lmark
        np.save(lmark_path[:-10] +'_%05d_diff.npy'%(min_index) , diff)
        datalist[index].append(min_index) 

    with open(os.path.join(root_path, 'pickle','train_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(datalist, handle, protocol=pkl.HIGHEST_PROTOCOL)
# swith_identity('turing1')
# swith_identity_obama()
# get_front()
# pca_lmark_grid()
# deepspeech_grid()
# pca_3dlmark_grid()
# data = np.load('/home/cxu-serve/p1/common/grid/align/s1/lwae8n_front.npy')[:,:,:2]
# data = data.reshape(data.shape[0], 136)
# print (data.shape)
# mean = np.load('../basics/mean_grid_front.npy')
# component = np.load('../basics/U_grid_front.npy')
# data_reduced = np.dot(data - mean, component.T)

# data_original = np.dot(data_reduced,component) + mean
# np.save( 'gg.npy', data_original )
# print (data - data_original)
# landmark_extractor(path = '/home/cxu-serve/p1/common/demo/oppo_demo')
# landmark_extractor(video_path = '/home/cxu-serve/p1/common/demo/oppo_demo/ouyang.mp4')
# RT_compute(video_path = '/home/cxu-serve/p1/common/demo/oppo_demo/ouyang.mp4')
get_front_video(video_path =  '/home/cxu-serve/p1/common/demo/oppo_demo/957_crop.mp4')
get_front_video(video_path =  '/home/cxu-serve/p1/common/demo/oppo_demo/958_crop.mp4')

get_front_video(video_path =  '/home/cxu-serve/p1/common/demo/oppo_demo/959_crop.mp4')
get_front_video(video_path =  '/home/cxu-serve/p1/common/demo/oppo_demo/ouyang_crop.mp4')


# diff()
# landmark_extractor()


