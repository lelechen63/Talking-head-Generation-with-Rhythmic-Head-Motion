import os
import shutil
import pdb
from tqdm import tqdm

data_root = '/home/cxu-serve/p1/common/experiment'
eva_root = '/home/cxu-serve/p1/common/evaluation'
gt_root = '/home/cxu-serve/p1/common/voxceleb2/unzip/test_video'

# dataset 
rid_sets = ['lrs', 'lrw', 'nips', 'few_shot', 'X2face']
datasets = os.listdir(data_root)
for dataset in tqdm(datasets):
    if dataset in rid_sets:
        continue

    test_modes = os.listdir(os.path.join(data_root, dataset))
    if not os.path.exists(os.path.join(eva_root, dataset)):
        os.makedirs(os.path.join(eva_root, dataset))
    if not os.path.exists(os.path.join(eva_root, "gt")):
        os.makedirs(os.path.join(eva_root, 'gt'))
    # test mode
    for test_mode in tqdm(test_modes):
        data_names = os.listdir(os.path.join(data_root, dataset, test_mode))
        if not os.path.exists(os.path.join(eva_root, dataset, test_mode)):
            os.makedirs(os.path.join(eva_root, dataset, test_mode))
        if not os.path.exists(os.path.join(eva_root, "gt", test_mode)):
            os.makedirs(os.path.join(eva_root, "gt", test_mode))
        # data name
        for data in data_names:
            try:
                source_video = os.path.join(data_root, dataset, test_mode, data, 'test.mp4')
                video_ = 'test'
                assert os.path.exists(source_video)
            except:
                source_video = os.path.join(data_root, dataset, test_mode, data, '{}.mp4'.format(data))
                video_ = data
                assert os.path.exists(source_video)
            dest_dir = os.path.join(eva_root, dataset, test_mode)

            shutil.copy(source_video, dest_dir)
            # dest_video = os.path.join(dest_dir, "{}.mp4".format(data))
            os.rename(os.path.join(dest_dir, "{}.mp4".format(video_)), \
                      os.path.join(dest_dir, "{}_fake.mp4".format(data)))

            # ground truch
            data_id = data.split('_')[0]
            video_name = data[8:-14]
            video_id = '{}_aligned.mp4'.format(data.split('_')[-2])
            if not os.path.exists(os.path.join(eva_root, "gt", test_mode,"{}_real.mp4".format(data))):
                shutil.copy(os.path.join(gt_root, data_id, video_name, video_id), \
                            os.path.join(eva_root, "gt", test_mode))
                os.rename(os.path.join(eva_root, "gt", test_mode, video_id), \
                          os.path.join(eva_root, "gt", test_mode,"{}_real.mp4".format(data)))
