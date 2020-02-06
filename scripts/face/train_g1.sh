# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt

# CUDA_VISIBLE_DEVICES=1 python train.py --name face8_linear --dataset_mode fewshot_face \
# --adaptive_spade --warp_ref \
# --gpu_ids 0 --batchSize 6 --nThreads 4 --niter 50 \
# --n_shot 1 \
# --dataroot '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped/train_file' \

CUDA_VISIBLE_DEVICES=1 python train.py --name face1_newloader --dataset_mode facefore \
--adaptive_spade --warp_ref \
--gpu_ids 0 --batchSize 2 --nThreads 0 --niter 10000 --niter_single 10001 \
--n_shot 8 --save_epoch_freq 50 --display_freq 5000 \
--dataroot '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/'