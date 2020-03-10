import os
import pdb

def add_audio(video_name, audio_dir):
    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
    #ffmpeg -i /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/new/audio/obama.wav -codec copy -c:v libx264 -c:a aac -b:a 192k  -shortest -y /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mov
    # ffmpeg -i gan_r_high_fake.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/audio/obama.wav -vcodec copy  -acodec copy -y   gan_r_high_fake.mov

    print (command)
    os.system(command)

def image_to_video(sample_dir = None, video_name = None):
    
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.jpg -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 
    #ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
    print (command)
    os.system(command)

def main():
    root = "/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/evaluation_store_good_retest/lrs"
    audio_root = "/home/cxu-serve/p1/common/lrs3/lrs3_v0.4"
    test_modes = os.listdir(root)
    for test_mode in test_modes:
        datasets = os.listdir(os.path.join(root, test_mode))
        for dataset in datasets:
            path = os.path.join(root, test_mode, dataset)
            # pdb.set_trace()
            image_to_video(path, os.path.join(path, '{}.mp4'.format(dataset)))
            add_audio(os.path.join(path, '{}.mp4'.format(dataset)), os.path.join(audio_root, 'test', dataset[5:-11], '{}.wav'.format(dataset.split('_')[-2])))

if __name__ == '__main__':
    main()