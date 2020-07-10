from PIL import Image
import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import pickle as pkl
from scipy.io import wavfile
from pydub import AudioSegment
from subprocess import check_call
import matplotlib.pyplot as plt
import math
import pdb
# import librosa
import glob
from tqdm import tqdm

import pdb
import utils


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.L = self.I = self.Lr = self.Ir = None
        self.n_frames_total = 1  # current number of frames to train in a single iteration

    def name(self):
        return 'BaseDataset'

    def update_training_batch(self, ratio):
        # update the training sequence length to be longer
        seq_len_max = 30
        if self.n_frames_total < seq_len_max:
            self.n_frames_total = min(
                seq_len_max, self.opt.n_frames_total * (2**ratio))
            print('--- Updating training sequence length to %d ---' %
                  self.n_frames_total)

    def read_data(self, path, data_type='img'):
        is_img = data_type == 'img'
        if is_img:
            img = Image.open(path)
        elif data_type == 'np':
            img = np.loadtxt(path, delimiter=',')
        elif data_type == 'npy':
            img = np.load(path)
        else:
            img = path
        return img

    def crop(self, img, coords):
        min_y, max_y, min_x, max_x = coords
        if isinstance(img, np.ndarray):
            return img[min_y:max_y, min_x:max_x]
        else:
            return img.crop((min_x, min_y, max_x, max_y))

    def concat_frame(self, A, Ai, ref=False):
        # if not self.opt.isTrain:
        #     return Ai
        if not ref and not self.opt.isTrain:
            return Ai

        if A is None:
            A = Ai
        else:
            A = torch.cat([A, Ai])
        return A

def m4a2mav():
    def worker(video_f):
        result = check_call(['ffmpeg', '-i', video_f, video_f[:-4]+'.wav'])
        print(result)

    files = glob.glob(os.path.join('/home/cxu-serve/p1/common/voxceleb2/unzip/test_audio/', '*', '*', '*.m4a'))
    for f in files:
        worker(f)

class VoxLmark2rgbDataset(BaseDataset):

    """ Dataset object used to access the pre-processed VoxCelebDataset """

    def __init__(self, opt, mode):
        self.opt = opt
        self.mode = mode

        # pdb.set_trace()
        if self.mode == 'train':
            _file = open(os.path.join(
                self.opt.dataroot, 'pickle', 'dev_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
            # pdb.set_trace()
            # ids = set([d[0] for d in self.data])
            # for id in tqdm(ids):
            #     if len([d[0] for d in self.data if d[0]==id]) > 1000:
            #         print(id)
            #     # print('id:{} num:{}'.format(id, ))
            # self.data = [d for d in self.data if d[0]=='id02833']
        elif self.mode == 'val':
            _file = open(os.path.join(self.opt.dataroot, 'pickle',
                                      'test_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            self.data = random.sample(self.data, opt.batch_size)
            _file.close()
        else:
            _file = open(os.path.join(self.opt.dataroot, 'pickle',
                                      'test_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()

        if self.mode == 'train':
            self.video_bag = 'unzip/dev_video'
        else:
            self.video_bag = 'unzip/test_video'

    def __len__(self):
        return len(self.data)

    def name(self):
        return 'VoxLmark2rgbDataset'

    def __getitem__(self, index):
        paths = self.data[index]
        rt_path = os.path.join(
            self.opt.dataroot, self.video_bag, paths[0], paths[1], paths[2]+"_aligned_rt.npy")
        audio_path = os.path.join(
            self.opt.dataroot, 'unzip/dev_audio' if self.mode == 'train' else 'unzip/test_audio', paths[0], paths[1], paths[2]) + '.wav'
        if not os.path.exists(audio_path):
            return self.__getitem__(random.choice(range(len(self.data))))

        ani_id = paths[3]
        rt = np.load(rt_path)
        fs, audio = self.parse_audio(audio_path, rt.shape[0])

        rt_mean = np.array([0.07, -0.008, -0.016, 11.4, 5.69, -4.37])
        rt_std = np.array([0.066, 0.132, 0.071, 9.99, 9.24, 16.95])
        rt = (rt-rt_mean)/rt_std

        for i in range(rt.shape[1]):
            rt[:, i] = utils.smooth(rt[:, i])
        for i in range(audio.shape[1]):
            for j in range(audio.shape[2]):
                audio[:,i,j] = utils.smooth(audio[:,i,j], 21)

        # audio = audio[:,3,:].mean(-1)
        # audio = utils.smooth(audio, 21)
        # pdb.set_trace()
        # plt.plot(range(rt.shape[0]), rt[:,0], label='rt_0')
        # plt.plot(range(audio.shape[0]), audio, label='audio_s')
        # plt.legend()
        # plt.savefig('/u/zkou2/Code/Trans/vs.png')
        # plt.close()

        return torch.from_numpy(rt), torch.from_numpy(audio)

    def parse_audio(self, path, frames):
        # audio = AudioSegment.from_file(path, format='raw', channels=1, sample_width=1, frame_rate=50000)
        # fs, mfcc = 50000, np.array(audio.get_array_of_samples())
        fs, mfcc = wavfile.read(path)
        assert fs == 16000
        # pdb.set_trace()
        # frame_len = 16000 // 25
        # frames = mfcc.shape[0]//frame_len
        # if frames*frame_len < mfcc.shape[0]:
        #     frames += 1
        #     mfcc = np.append(mfcc, np.zeros(frames*frame_len-mfcc.shape[0]), 0)

        chunck_size = int(fs * 0.04)
        frame_len = chunck_size
        left_append = mfcc[: 3 * chunck_size]
        right_append = mfcc[-4 * chunck_size:]
        mfcc = np.insert(mfcc, 0, left_append, axis=0)
        mfcc = np.insert(mfcc, -1, right_append, axis=0)
        res = []
        # mfcc = utils.smooth(mfcc, 21)
        for i in range(frames):
            res.append(mfcc[i*frame_len:i*frame_len+7*frame_len])

        res = np.stack(res)
        res = res.reshape(res.shape[0], -1, frame_len)
        return fs, res

if __name__ == '__main__':
    m4a2mav()