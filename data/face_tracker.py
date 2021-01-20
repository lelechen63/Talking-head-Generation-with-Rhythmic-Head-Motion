from collections import OrderedDict
import tempfile
import shutil
import numpy as np 
import mmcv
import scipy.ndimage.morphology
import cv2 
import dlib
import face_alignment
from scipy.spatial.transform import Rotation 
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from tqdm import tqdm
import time
import  utils.util as util
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
res = 224
import  utils.visualizer as Visualizer
import pickle
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)#,  device='cpu' )

def crop_image(frame_file, count =0, x_list = [], y_list = [], dis_list = [], videos = [],multi_face_times = 0, lStart=36, lEnd=41, rStart=42, rEnd=47):
	image = cv2.imread(frame_file) if isinstance(frame_file, str) else frame_file
	
	videos.append(image)
	new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	preds = fa.get_landmarks(new_gray)
	# if len(preds) > 2: 
	# 	print (count, len(preds))
	# 	multi_face_times += 1
	# 	if multi_face_times > 2:
	# 		print ('multiple faces')
	# 		raise Exception('multiple faces')

	new_shape = preds[0]
	leftEyePts = new_shape[lStart:lEnd]
	rightEyePts = new_shape[rStart:rEnd]

	leftEyeCenter = leftEyePts.mean(axis=0)
	rightEyeCenter = rightEyePts.mean(axis=0)
	max_v = np.amax(new_shape, axis=0)
	min_v = np.amin(new_shape, axis=0)

	max_x, max_y = max_v[0], max_v[1]
	min_x, min_y = min_v[0], min_v[1]
	dis = max(max_y - min_y, max_x - min_x)

	two_eye_center = (leftEyeCenter + rightEyeCenter)/2
	center_y, center_x = two_eye_center[0], two_eye_center[1]
	x_list =np.append( x_list, center_x )
	y_list = np.append(y_list, center_y)
	dis_list = np.append(dis_list, dis)
	return  x_list, y_list, dis_list, videos, multi_face_times



def crop_face_region_image(frame_file, count = 0, x_list = [], y_list = [], dis_list = [], videos = [],multi_face_times = 0, lStart=36, lEnd=41, rStart=42, rEnd=47):
	image = cv2.imread(frame_file) if isinstance(frame_file, str) else frame_file
	
	videos.append(image)
	new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	preds = fa.get_landmarks(new_gray)
	new_shape = preds[0]
	leftEyePts = new_shape[lStart:lEnd]
	rightEyePts = new_shape[rStart:rEnd]

	leftEyeCenter = leftEyePts.mean(axis=0)
	rightEyeCenter = rightEyePts.mean(axis=0)
	max_v = np.amax(new_shape, axis=0)
	min_v = np.amin(new_shape, axis=0)

	max_x, max_y = max_v[0], max_v[1]
	min_x, min_y = min_v[0], min_v[1]
	dis = max(max_y - min_y, max_x - min_x)

	two_eye_center = (leftEyeCenter + rightEyeCenter)/2
	center_y, center_x = two_eye_center[0], two_eye_center[1]
	x_list =np.append( x_list, center_x )
	y_list = np.append(y_list, center_y)
	dis_list = np.append(dis_list, dis)
	return  x_list, y_list, dis_list, videos, multi_face_times

def _crop_face_region_video(video):
	cap  =  cv2.VideoCapture(video)
	count = 0
	x_list =  np.array([])
	y_list = np.array([])
	dis_list = np.array([])
	videos = []
	multi_face_times = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		
		if ret == True:
			x_list, y_list, dis_list, videos, multi_face_times = crop_image(frame, count, x_list, y_list, dis_list, videos, multi_face_times)
			count += 1
		else:
			break
	dis = np.mean(dis_list)

	top_left_x = x_list - (40 * dis / 90)
	top_left_y = y_list - (50* dis / 90)
	top_left_x = util.oned_smooth(top_left_x )
	top_left_y = util.oned_smooth(top_left_y)	
	side_length = int((100 * dis / 90))
	# dirpath = tempfile.mkdtemp()
	dirpath = './gg'
	for i in tqdm(range(x_list.shape[0])):
		if top_left_x[i] < 0 or top_left_y[i] < 0:
			img_size = videos[i].shape
			tempolate = np.zeros((img_size[0] * 2, img_size[1]* 2 , 3), np.uint8) 
			tempolate_middle  = [int(tempolate.shape[0]/2), int(tempolate.shape[1]/2)]
			middle = [int(img_size[0]/2), int(img_size[1]/2)]
			tempolate[tempolate_middle[0]  -middle[0]:tempolate_middle[0]+middle[0], tempolate_middle[1]-middle[1]:tempolate_middle[1]+middle[1], :] = videos[i]
			top_left_x[i] = top_left_x[i] + tempolate_middle[0]  -middle[0]
			top_left_y[i] = top_left_y[i] + tempolate_middle[1]  -middle[1]
			roi = tempolate[int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
			roi =cv2.resize(roi,(256,256))
			cv2.imwrite(  os.path.join(dirpath,'%05d.png'%i ) , roi)

		else:
			roi = videos[i][int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
			roi =cv2.resize(roi,(256,256))
			cv2.imwrite( os.path.join(dirpath,'%05d.png'%i ) , roi)

	# command = 'ffmpeg -framerate 25  -i ' +   './temp/%05d.png  -vcodec libx264 -y -vf format=yuv420p ' + video.replace('raw', 'cropped') 
	# os.system(command)
	# shutil.rmtree(dirpath)
	print (dirpath)

# _crop_face_region_video('/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped/videos/000.mp4')


def _crop_img(img_path, pid = 0, mode = 0):
	frame = cv2.imread(img_path)
		
	x_list, y_list, dis_list, videos, multi_face_times = crop_image(frame, 0, [], [], [], [], 0)

	dis = np.mean(dis_list)
	print (dis)
	side_length = int((205 * dis / 90))
	print (side_length)
	
	top_left_x = x_list - (80 * dis / 90)
	top_left_y = y_list - (100* dis / 90)
	top_left_x_mean = np.mean(top_left_x)

	top_left_y_mean = np.mean(top_left_y)
	print (top_left_x_mean, top_left_y_mean)
	for g in range(len(top_left_x)):
		top_left_x[g] = top_left_x_mean

		top_left_y[g] = top_left_y_mean


	i = 0
	if top_left_x[i] < 0 or top_left_y[i] < 0:
		img_size = videos[i].shape
		tempolate = np.zeros((img_size[0] * 2, img_size[1]* 2 , 3), np.uint8)
		tempolate_middle  = [int(tempolate.shape[0]/2), int(tempolate.shape[1]/2)]
		middle = [int(img_size[0]/2), int(img_size[1]/2)]
		tempolate[tempolate_middle[0]  -middle[0]:tempolate_middle[0]+middle[0], tempolate_middle[1]-middle[1]:tempolate_middle[1]+middle[1], :] = videos[i]
		top_left_x[i] = top_left_x[i] + tempolate_middle[0]  -middle[0]
		top_left_y[i] = top_left_y[i] + tempolate_middle[1]  -middle[1]
		roi = tempolate[int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
		roi =cv2.resize(roi,(256,256))
		cv2.imwrite(img_path[:-4] +'_crop.png', roi)
	else:
		roi = videos[i][int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
		roi =cv2.resize(roi,(256,256))
		cv2.imwrite(img_path[:-4] +'_crop.png', roi)

	return roi
def _crop_video(video, pid = 0, mode = 0):
	count = 0
	x_list =  np.array([])
	y_list = np.array([])
	dis_list = np.array([])
	videos = []
	cap  =  cv2.VideoCapture(video)
	multi_face_times = 0
	if os.path.exists('./temp%05d'%pid):
		shutil.rmtree('./temp%05d'%pid)
	os.mkdir('./temp%05d'%pid)
	while(cap.isOpened()):
		ret, frame = cap.read()
		# if count == 20:
		# 	break
		if ret == True:
			x_list, y_list, dis_list, videos, multi_face_times = crop_image(frame, count, x_list, y_list, dis_list, videos, multi_face_times)
			count += 1
		else:
			break
	dis = np.mean(dis_list)
	print (dis)
	side_length = int((205 * dis / 90))
	print (side_length)
	if mode ==0 :
		top_left_x = x_list - (80 * dis / 90)
		top_left_y = y_list - (100* dis / 90)
		top_left_x = util.oned_smooth(top_left_x )
		top_left_y = util.oned_smooth(top_left_y)	
		
	else:
		top_left_x = x_list - (80 * dis / 90)
		top_left_y = y_list - (100* dis / 90)
		top_left_x_mean = np.mean(top_left_x)

		top_left_y_mean = np.mean(top_left_y)
		print (top_left_x_mean, top_left_y_mean)
		for g in range(len(top_left_x)):
			top_left_x[g] = top_left_x_mean

			top_left_y[g] = top_left_y_mean


	for i in tqdm(range(x_list.shape[0])):
		if top_left_x[i] < 0 or top_left_y[i] < 0:
			img_size = videos[i].shape
			tempolate = np.zeros((img_size[0] * 2, img_size[1]* 2 , 3), np.uint8)
			tempolate_middle  = [int(tempolate.shape[0]/2), int(tempolate.shape[1]/2)]
			middle = [int(img_size[0]/2), int(img_size[1]/2)]
			tempolate[tempolate_middle[0]  -middle[0]:tempolate_middle[0]+middle[0], tempolate_middle[1]-middle[1]:tempolate_middle[1]+middle[1], :] = videos[i]
			top_left_x[i] = top_left_x[i] + tempolate_middle[0]  -middle[0]
			top_left_y[i] = top_left_y[i] + tempolate_middle[1]  -middle[1]
			roi = tempolate[int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
			roi =cv2.resize(roi,(256,256))
			cv2.imwrite('./temp%05d/%05d.png'%(pid, i), roi)
		else:
			roi = videos[i][int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
			roi =cv2.resize(roi,(256,256))
			cv2.imwrite('./temp%05d/%05d.png'%(pid, i), roi)

	# command = 'ffmpeg -framerate 25  -i ' +   './temp/%05d.png  -vcodec libx264 -y -vf format=yuv420p ' + video.replace('raw', 'cropped') 
	# os.system(command)

# _crop_video('/u/lchen63/lchen63_data/addition_example/id01822/00003.mp4')

# def face_forensics_dataset_crop_preprocess():
# 	datapath = '/u/lchen63/lchen63_data/faceforensics/original_sequences/youtube/raw/videos'
# 	pid = 0
# 	for i, v_name in enumerate(sorted(os.listdir(datapath))):  
# 		print (v_name)
# 		try:
# 			_crop_video( os.path.join(datapath , v_name), pid )
# 			command = 'ffmpeg -framerate 25  -i ' +   './temp%05d'%pid + '/%05d.png  -vcodec libx264 -y -vf format=yuv420p ' +  os.path.join(datapath , v_name).replace('raw', 'cropped') 
# 			os.system(command)
# 		except:
# 			print ( '===================')
# 			continue
# 		# if i == 3:
# 		# 	break
# # face_forensics_dataset_crop_preprocess()
# # face_forensics_dataset_audio_preprocess()
