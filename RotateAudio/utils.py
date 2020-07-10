import pdb
import torch
import os
import shutil
import numpy as np
import math
import cv2


def oned_smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.
    output:
            the smoothed signal

    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size."
               )

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (
            ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


class AdjustLR(object):
    def __init__(self, optimizer, init_lr, sleep_epochs=3, half=5, verbose=0):
        super(AdjustLR, self).__init__()
        self.optimizer = optimizer
        self.sleep_epochs = sleep_epochs
        self.half = half
        self.init_lr = init_lr
        self.verbose = verbose

    def step(self, epoch):
        if epoch >= self.sleep_epochs:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                new_lr = self.init_lr[idx] * \
                    math.pow(0.5, (epoch-self.sleep_epochs+1)/float(self.half))
                param_group['lr'] = new_lr
            if self.verbose:
                print('>>> reduce learning rate <<<')


def get_model(options):
    # Choose the embedding network
    if options.rt_encode == 'tc':
        from models.rt_encode import TemporalConv
        rt_encode = TemporalConv(options)
    elif options.rt_encode == 'tc_no_lstm':
        from models.rt_encode import TemporalConv_no_LSTM
        rt_encode = TemporalConv_no_LSTM(options)
    elif options.rt_encode == 'rt_lstm':
        from models.rt_encode import RT_LSTM
        rt_encode = RT_LSTM(options)
    else:
        raise

    if options.audio_encode == 'a2l':
        from models.audio_encode import A2L
        audio_encode = A2L(options)
    elif options.audio_encode == 'se':
        from models.audio_encode import SingleEmb
        audio_encode = SingleEmb(options)
    elif options.audio_encode == 'oe':
        from models.audio_encode import OneEmb
        audio_encode = OneEmb(options)
    else:
        raise

    if options.rt_decode == 'lstm':
        from models.rt_decode import LSTMDecode
        rt_decode = LSTMDecode(options)
    elif options.rt_decode == 'linear':
        from models.rt_decode import LinearDecode
        rt_decode = LinearDecode(options)
    elif options.rt_decode == 'l2l':
        from models.rt_decode import L2LDecode
        rt_decode = L2LDecode(options)
    elif options.rt_decode == 'rt_lstm':
        from models.rt_decode import RTDecode
        rt_decode = RTDecode(options)
    elif options.rt_decode == 'linear_no_lstm':
        from models.rt_decode import LinearDecode_no_LSTM
        rt_decode = LinearDecode_no_LSTM(options)
    else:
        raise

    return rt_encode, audio_encode, rt_decode


def onehot(indices, depth):
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies.type(torch.LongTensor)


def init_log_dir(opt):
    if os.path.exists(os.path.join('./save', opt.name)):
        print('dir exist, delete?')
        x = input()
        if x == 'y':
            shutil.rmtree(os.path.join('./save', opt.name))
        else:
            raise

    os.mkdir(os.path.join('./save', opt.name))
    with open(os.path.join('./save', opt.name, 'options.txt'), "a") as f:
        for k, v in vars(opt).items():
            f.write('{} -> {}\n'.format(k, v))

    os.mkdir(os.path.join('./save', opt.name, 'check'))
    os.mkdir(os.path.join('./save', opt.name, 'imgs'))
    os.mkdir(os.path.join('./save', opt.name, 'tb'))


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.size(-1)  # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)


def get_imgs_from_video(video):
    frames = []
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise

    if x.size < window_len:
        raise


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2-1):-(window_len//2+1)]