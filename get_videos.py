import os
import shutil
import pdb
from tqdm import tqdm

# data_root = '/home/cxu-serve/p1/common/experiment'
data_root = './evaluation_store_good_retest'
eva_root = './final_result_store_lrs'

# dataset 
rid_sets = ['face8_vox_ani']
datasets = os.listdir(data_root)
for dataset in tqdm(datasets):
    if dataset in rid_sets:
        continue

    test_modes = os.listdir(os.path.join(data_root, dataset))
    if not os.path.exists(os.path.join(eva_root, dataset)):
        os.makedirs(os.path.join(eva_root, dataset))
    # test mode
    for test_mode in tqdm(test_modes):
        if not os.path.isdir(os.path.join(data_root, dataset, test_mode)):
            continue
        data_names = os.listdir(os.path.join(data_root, dataset, test_mode))
        if not os.path.exists(os.path.join(eva_root, dataset, test_mode)):
            os.makedirs(os.path.join(eva_root, dataset, test_mode))
        # data name
        for data in data_names:
            files = os.listdir(os.path.join(data_root, dataset, test_mode, data))
            source_videos = [f for f in files if f[-3:]=='mp4']
            # get videos
            # pdb.set_trace()
            for source_video in source_videos:
                dest_dir = os.path.join(eva_root, dataset, test_mode)
                source_video_dir = os.path.join(data_root, dataset, test_mode, data, source_video)
                shutil.copy(source_video_dir, dest_dir)
            
                os.rename(os.path.join(dest_dir, source_video), \
                        os.path.join(dest_dir, "{}_fake.mov".format(data)))