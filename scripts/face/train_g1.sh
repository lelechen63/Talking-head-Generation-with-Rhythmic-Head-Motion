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

# CUDA_VISIBLE_DEVICES=1 python train.py --name face8_face_newloader --dataset_mode facefore \
# --adaptive_spade --warp_ref \
# --gpu_ids 0 --batchSize 2 --nThreads 0 --niter 500 --niter_single 0 --niter_step 10 \
# --n_shot 8 --save_epoch_freq 50 --display_freq 20 \
# --n_frames_G 2 \
# --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid --continue_train

# CUDA_VISIBLE_DEVICES=3 python train.py --name face8_face_newloader --dataset_mode facefore \
# --adaptive_spade --warp_ref --warp_ani \
# --gpu_ids 0 --batchSize 2 --nThreads 0 --niter 500 --niter_single 501 --niter_step 1 \
# --n_shot 8 --save_epoch_freq 50 \
# --n_frames_G 1 --add_raw_loss \
# --dataroot '/home/cxu-serve/p1/common/voxceleb2' --dataset_name vox

# CUDA_VISIBLE_DEVICES=2 python train.py --name face8_face_newloader --dataset_mode facefore \
# --adaptive_spade --warp_ref \
# --gpu_ids 0 --batchSize 2 --nThreads 0 --niter 1000 --niter_single 1001 \
# --n_shot 1 --n_frames_G 1 --ref_ratio 0 \
# --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid \
# --continue_train

# CUDA_VISIBLE_DEVICES=3 python train.py --name face8_face_newloader --dataset_mode facefore \
# --adaptive_spade --warp_ref --spade_combine \
# --gpu_ids 0 --batchSize 1 --nThreads 8 --niter 1000 --niter_single 1001 \
# --n_shot 1 --n_frames_G 1 --ref_ratio 0 --serial_batches \
# --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid \
# --continue_train

# CUDA_VISIBLE_DEVICES=2 python train.py --name face8_crema_linear --dataset_mode facefore \
# --adaptive_spade --warp_ref \
# --gpu_ids 0 --batchSize 2 --nThreads 0 --niter 500 --niter_single 501 \
# --n_shot 8 --save_epoch_freq 50 --display_freq 1 \
# --n_frames_G 1 \
# --dataroot '/home/cxu-serve/p1/common/CREMA' --dataset_name crema --continue_train

# CUDA_VISIBLE_DEVICES=2 python train.py --name face8_vox_ani_nonlinear_comp --dataset_mode facefore \
# --adaptive_spade --warp_ref --spade_combine --add_raw_loss \
# --gpu_ids 0 --batchSize 2 --nThreads 8 --niter 1000 --niter_single 1001 \
# --n_shot 8 --n_frames_G 1 \
# --dataroot '/home/cxu-serve/p1/common/voxceleb2' --dataset_name vox --save_epoch_freq 2 --display_freq 1 \
# --continue_train

# CUDA_VISIBLE_DEVICES=1 python train.py --name face8_vox_new_nonlinear --dataset_mode facefore \
# --adaptive_spade --warp_ref --warp_ani --spade_combine --add_raw_loss \
# --gpu_ids 0 --batchSize 2 --nThreads 0 --niter 1000 --niter_single 1001 \
# --n_shot 8 --n_frames_G 1 \
# --dataroot '/home/cxu-serve/p1/common/voxceleb2' --dataset_name vox --save_epoch_freq 1 --display_freq 1 \
# --continue_train --no_warp

# CUDA_VISIBLE_DEVICES=1 python train.py --name face8_vox_new_nonlinear --dataset_mode facefore \
# --adaptive_spade --warp_ref --warp_ani --spade_combine --add_raw_loss \
# --gpu_ids 0 --batchSize 2 --nThreads 0 --niter 1000 --niter_single 1001 \
# --n_shot 8 --n_frames_G 1 \
# --dataroot '/home/cxu-serve/p1/common/voxceleb2' --dataset_name vox --save_epoch_freq 1 --display_freq 1 \
# --continue_train --crop_ref

# CUDA_VISIBLE_DEVICES=1 python train.py --name face8_vox_new_nonlinear --dataset_mode facefore \
# --adaptive_spade --warp_ref --warp_ani --spade_combine --add_raw_loss \
# --gpu_ids 0 --batchSize 2 --nThreads 0 --niter 1000 --niter_single 1001 \
# --n_shot 8 --n_frames_G 1 \
# --dataroot '/home/cxu-serve/p1/common/voxceleb2' --dataset_name vox --save_epoch_freq 1 --display_freq 1 \
# --continue_train --no_atten

# CUDA_VISIBLE_DEVICES=0 python train.py --name face8_crema_mouth_nonlinear_L1 --dataset_mode facefore \
# --adaptive_spade --warp_ref --spade_combine --add_raw_loss \
# --gpu_ids 0 --batchSize 2 --nThreads 0 --niter 50 --niter_single 51 \
# --n_shot 8 --n_frames_G 1 --ref_ratio 0 \
# --dataroot '/home/cxu-serve/p1/common/CREMA' --dataset_name crema --save_epoch_freq 1 \
# --continue_train --crop_ref --find_largest_mouth

CUDA_VISIBLE_DEVICES=0 python train.py --name face8_vox_test --dataset_mode facefore \
--adaptive_spade --warp_ref --spade_combine --add_raw_loss \
--gpu_ids 0 --batchSize 2 --nThreads 0 --niter 20 --niter_single 21 \
--n_shot 8 --n_frames_G 1 \
--lambda_flow 0.2 --lambda_vgg 2 --lambda_mouth_vgg 1 \
--dataroot '/home/cxu-serve/p1/common/voxceleb2' --dataset_name vox --save_epoch_freq 2 --display_freq 1000 \
--which_model_netD 'syncframe' \
--continue_train --crop_ref --audio_drive --add_mouth_D --use_new_D