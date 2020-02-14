# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt

# python train.py --name face --dataset_mode fewshot_face \
# --adaptive_spade --warp_ref --spade_combine \
# --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 60 --nThreads 16 --continue_train

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --name face8_vox_linear --dataset_mode facefore \
# --adaptive_spade --warp_ref \
# --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 80 --nThreads 64 --niter 1000 --niter_single 1001 \
# --n_shot 8 --n_frames_G 1 \
# --dataroot '/mnt/Data/lchen63/voxceleb2' --dataset_name vox

# CUDA_VISIBLE_DEVICES=0,1,2 python train.py --name face8_face_linear --dataset_mode facefore \
# --adaptive_spade --warp_ref \
# --gpu_ids 0,1,2 --batchSize 9 --nThreads 8 --niter 500 --niter_single 501 \
# --n_shot 8 \
# --dataroot '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube' \
# --dataset_name face --continue_train

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name face8_vox_ani --dataset_mode facefore \
--adaptive_spade --warp_ref --warp_ani \
--gpu_ids 0,1,2,3 --batchSize 8 --nThreads 8 --niter 1000 --niter_single 1001 \
--n_shot 8 --save_epoch_freq 50 \
--n_frames_G 1 \
--dataroot '/home/cxu-serve/p1/common/voxceleb2' --dataset_name vox \
--continue_train
