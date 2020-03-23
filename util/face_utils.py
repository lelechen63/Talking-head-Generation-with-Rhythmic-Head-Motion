import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib.lines as mlines
from matplotlib import transforms
import argparse, os, fnmatch, shutil
import numpy as np
import cv2
import math
import copy
import librosa
import subprocess
from tqdm import tqdm
import tempfile
from . import util
font = {'size'   : 18}
mpl.rc('font', **font)

# Lookup tables for drawing lines between points
Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
         [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
         [66, 67], [67, 60]]

Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33], \
        [33, 34], [34, 35], [27, 31], [27, 35]]

leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]

leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
         [12, 13], [13, 14], [14, 15], [15, 16]]

faceLmarkLookup = Mouth + Nose + leftBrow + rightBrow + leftEye + rightEye + other

def mounth_open2close(lmark): # if the open rate is too large, we need to manually make the mounth to be closed.
    # the input lamrk need to be (68,2 ) or (68,3)
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    upper_part = [49,50,51,52,53]
    lower_part = [59,58,57,56,55]
    diffs = []

    for k in range(3):
        mean = (lmark[open_pair[k][0],:2] + lmark[open_pair[k][1],:2] )/ 2
        print (mean)
        tmp = lmark[open_pair[k][0],:2]
        diffs.append((mean - lmark[open_pair[k][0],:2]).copy())
        lmark[open_pair[k][0],:2] = mean - (mean - lmark[open_pair[k][0],:2]) * 0.3
        lmark[open_pair[k][1],:2] = mean + (mean - lmark[open_pair[k][0],:2]) * 0.3
    diffs.insert(0, 0.6 * diffs[2])
    diffs.append( 0.6 * diffs[2])
    print (diffs)
    diffs = np.asarray(diffs)
    lmark[49:54,:2] +=  diffs
    lmark[55:60,:2] -=  diffs 
    return lmark



def get_roi(lmark):
    tempolate = np.zeros((256, 256 , 3), np.uint8)
    eyes =[17, 20 , 21, 22, 24,  26, 36, 39,42, 45]
    eyes_x = []
    eyes_y = []
    for i in eyes:
        eyes_x.append(lmark[i,0])
        eyes_y.append(lmark[i,1])
    min_x = lmark[eyes[np.argmin(eyes_x)], 0] 
    max_x = lmark[eyes[np.argmax(eyes_x)], 0] 
    min_y = lmark[eyes[np.argmin(eyes_y)], 1]
    
    max_y = lmark[eyes[np.argmax(eyes_y)], 1]
    min_x = max(0, int(min_x-10) )
    max_x = min(255, int(max_x+10) )
    min_y = max(0, int(min_y-10) )
    max_y = min(255, int(max_y+10) )

    tempolate[ int(min_y): int(max_y), int(min_x):int(max_x)] = 1 
    mouth = [48, 50, 51, 54, 57]
    mouth_x = []
    mouth_y = []
    for i in mouth:
        mouth_x.append(lmark[i,0])
        mouth_y.append(lmark[i,1])
    min_x2 = lmark[mouth[np.argmin(mouth_x)], 0] 
    max_x2 = lmark[mouth[np.argmax(mouth_x)], 0] 
    min_y2 = lmark[mouth[np.argmin(mouth_y)], 1]
    max_y2 = lmark[mouth[np.argmax(mouth_y)], 1] 

    min_x2 = max(0, int(min_x2-10) )
    max_x2 = min(255, int(max_x2+10) )
    min_y2 = max(0, int(min_y2-10) )
    max_y2 = min(255, int(max_y2+10) )

    
    tempolate[int(min_y2):int(max_y2), int(min_x2):int(max_x2)] = 1
    return  tempolate


def eye_blinking(lmark, rate = 40): #lmark shape (k, 68,2) or (k,68,3) , tempolate shape(256, 256, 1)
    length = lmark.shape[0]
    bink_time = math.floor(length / float(rate) )
    
    eys =[[37,41],[38,40] ,[43,47],[44,46]]  # [upper, lower] , [left1,left2, right1, right1]
    
    for i in range(bink_time):

        print ('+++++')
        for e in eys:
            dis =  (np.abs(lmark[0, e[0],:2] -  lmark[0, e[1],:2] ) / 2)
            print ('--------')
            # -2 
            
            lmark[rate * (i + 1)-2, e[0],:2] += 0.45 * (dis)
            lmark[rate * (i + 1)-2, e[1],:2] -= 0.45 * (dis)
            # +2
            lmark[rate * (i + 1)+2, e[0], :2] += 0.45 * (dis)
            lmark[rate * (i + 1)+2, e[1], :2] -= 0.45 * (dis)

            # -1
            lmark[rate * (i + 1)-1, e[0], :2] += 0.85 * (dis)
            lmark[rate * (i + 1)-1, e[1], :2] -= 0.85 * (dis)
            # +1
            lmark[rate * (i + 1)+1, e[0], :2] += 0.8 * (dis)
            lmark[rate * (i + 1)+1, e[1], :2] -= 0.8 * (dis)

            # 0
            lmark[rate * (i + 1), e[0], :2] += 0.95 * (dis)
            lmark[rate * (i + 1), e[1], :2] -= 0.95 * (dis)
    return lmark

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T
    
    t = -R*centroid_A.T + centroid_B.T

    return R, t
class faceNormalizer(object):
    # Credits: http://www.learnopencv.com/face-morph-using-opencv-cpp-python/
    w = 256
    h = 256

    def __init__(self, w = 256, h = 256):
        self.w = w
        self.h = h

    def similarityTransform(self, inPoints, outPoints):
        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)
      
        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()
        
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
        
        inPts.append([np.int(xin), np.int(yin)])
        
        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
        
        outPts.append([np.int(xout), np.int(yout)])
        
        tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
        
        return tform

    def tformFlmarks(self, flmark, tform):
        transformed = np.reshape(np.array(flmark), (68, 1, 2))           
        transformed = cv2.transform(transformed, tform)
        transformed = np.float32(np.reshape(transformed, (68, 2)))
        return transformed

    def alignEyePoints(self, lmarkSeq):
        w = self.w
        h = self.h

        alignedSeq = copy.deepcopy(lmarkSeq)
        firstFlmark = alignedSeq[0,:,:]
        
        eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]
        eyecornerSrc  = [ (firstFlmark[36, 0], firstFlmark[36, 1]), (firstFlmark[45, 0], firstFlmark[45, 1]) ]

        tform = self.similarityTransform(eyecornerSrc, eyecornerDst)

        for i, lmark in enumerate(alignedSeq):
            alignedSeq[i] = self.tformFlmarks(lmark, tform)

        return alignedSeq

    def alignEyePointsV2(self, lmarkSeq):
        w = self.w
        h = self.h

        alignedSeq = copy.deepcopy(lmarkSeq)
        
        eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]
    
        for i, lmark in enumerate(alignedSeq):
            curLmark = alignedSeq[i,:,:]
            eyecornerSrc  = [ (curLmark[36, 0], curLmark[36, 1]), (curLmark[45, 0], curLmark[45, 1]) ]
            tform = self.similarityTransform(eyecornerSrc, eyecornerDst)
            alignedSeq[i,:,:] = self.tformFlmarks(lmark, tform)

        return alignedSeq

    def transferExpression(self, lmarkSeq, meanShape):
        exptransSeq = copy.deepcopy(lmarkSeq)
        firstFlmark = exptransSeq[0,:,:]
        indexes = np.array([60, 64, 62, 67])
        
        tformMS = cv2.estimateRigidTransform(firstFlmark[:,:], np.float32(meanShape[:,:]) , True)

        sx = np.sign(tformMS[0,0])*np.sqrt(tformMS[0,0]**2 + tformMS[0,1]**2)
        sy = np.sign(tformMS[1,0])*np.sqrt(tformMS[1,0]**2 + tformMS[1,1]**2)
        print (sx, sy)
        prevLmark = copy.deepcopy(firstFlmark)
        prevExpTransFlmark = copy.deepcopy(meanShape)
        zeroVecD = np.zeros((1, 68, 2))
        diff = np.cumsum(np.insert(np.diff(exptransSeq, n=1, axis=0), 0, zeroVecD, axis=0), axis=0)
        msSeq = np.tile(np.reshape(meanShape, (1, 68, 2)), [lmarkSeq.shape[0], 1, 1])

        diff[:, :, 0] = abs(sx)*diff[:, :, 0]
        diff[:, :, 1] = abs(sy)*diff[:, :, 1]

        exptransSeq = diff + msSeq

        return exptransSeq

    def unitNorm(self, flmarkSeq):
        normSeq = copy.deepcopy(flmarkSeq)
        normSeq[:, : , 0] /= self.w
        normSeq[:, : , 1] /= self.h
        return normSeq

def smooth(x,window_len=11,window='hanning'):
   
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
def write_video_wpts_wsound_unnorm(frames, sound, fs, path, fname, xLim, yLim):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print ('Exp')

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]/2, 2))
    print (frames.shape)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=25, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l, = plt.plot([], [], 'ko', ms=4)


    plt.xlim(xLim)
    plt.ylim(yLim)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(dt))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        plt.gca().invert_yaxis()
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i,:,0], frames[i,:,1])
            cnt = 0
            for refpts in faceLmarkLookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_ws.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))

def write_video_wpts_wsound(frames, sound, fs, path, fname, xLim, yLim):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        print ('Exp')

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]/2, 2))
    print (frames.shape)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=25, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    l, = plt.plot([], [], 'ko', ms=4)


    plt.xlim(xLim)
    plt.ylim(yLim)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    rect = (0, 0, 600, 600)
    
    if frames.shape[1] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print (lookup)
    else:
        lookup = faceLmarkLookup

    lines = [plt.plot([], [], 'k')[0] for _ in range(3*len(lookup))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        plt.gca().invert_yaxis()
        for i in tqdm(range(frames.shape[0])):
            l.set_data(frames[i,:,0], frames[i,:,1])
            cnt = 0
            for refpts in lookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -y -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_ws.mp4'
    subprocess.call(cmd, shell=True) 
    print('Muxing Done')

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))


def plot_flmarks(pts, lab, xLim, yLim, xLab, yLab, figsize=(10, 10)):
    if len(pts.shape) != 3:
        pts = pts.reshape(68, 2)

    if pts.shape[0] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print (lookup)
    else:
        lookup = faceLmarkLookup

    plt.figure(figsize=figsize)
    plt.plot(pts[:,0], pts[:,1], 'ko', ms=4)
    for refpts in lookup:
        plt.plot([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]], 'k', ms=4)

    plt.xlabel(xLab, fontsize = font['size'] + 4, fontweight='bold')
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top') 
    plt.ylabel(yLab, fontsize = font['size'] + 4, fontweight='bold')
    plt.xlim(xLim)
    plt.ylim(yLim)
    plt.gca().invert_yaxis()
   
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()





def plot_lmark_as_video(pts, video_name ='./gg.mp4', audio = None,  xLim=(0.0, 256.0), yLim=(0.0, 256.0), xLab = 'x', yLab = 'y', figsize=(10, 10) ):
    # shape need to be (k, 68 ,3 ) or (k, 68 ,2 )  ans pts need to be numpy array, audio need to be a .wav file
    dirpath = tempfile.mkdtemp()

    for i in range(pts.shape[0]):
        name = os.path.join( dirpath , '%05d.png'%i)
        plot_flmarks(pts[i,:,:2], name, xLim, yLim, xLab, yLab)

    util.image_to_video(dirpath,video_name )
    if audio is not None:
        util.add_audio(video_name,audio )

        
    # shutil.rmtree(dirpath)



def melSpectra(y, sr, wsize, hsize):

    cnst = 1+(int(sr*wsize)/2)
    y_stft_abs = np.abs(librosa.stft(y,
                                  win_length = int(sr*wsize),
                                  hop_length = int(sr*hsize),
                                  n_fft=int(sr*wsize)))/cnst

    melspec = np.log(1e-16+librosa.feature.melspectrogram(sr=sr, 
                                             S=y_stft_abs**2,
                                             n_mels=64))
    return melspec

def main():
    return

if __name__ == "__main__":
    main()