# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt

# python train.py --name face --dataset_mode fewshot_face \
# --adaptive_spade --warp_ref --spade_combine \
# --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 60 --nThreads 16 --continue_train

train_vox_linear(){
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --name face8_vox_ani --dataset_mode facefore \
    --adaptive_spade --warp_ref --warp_ani \
    --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 72 --nThreads 64 --niter 1000 --niter_single 1001 \
    --n_shot 8 --n_frames_G 1 \
    --dataroot '/mnt/Data/lchen63/voxceleb2' --dataset_name vox --save_epoch_freq 50 --display_freq 5000 \
    --continue_train 
}

train_grid_linear(){
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --name face8_grid_ani_retrain --dataset_mode facefore \
    --adaptive_spade --warp_ref \
    --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 12 --nThreads 64 --niter 30 --niter_single 1001 \
    --n_shot 1 --n_frames_G 1 --ref_ratio 0 \
    --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid --save_epoch_freq 1 --display_freq 1000 \
    --continue_train
}

train_grid_linear_temp_newflow(){
    CUDA_VISIBLE_DEVICES=1,2,3 python train.py --name face8_grid_ani_retrain_temp_newflow --dataset_mode facefore \
    --adaptive_spade --warp_ref \
    --gpu_ids 0,1,2 --batchSize 12 --nThreads 8 --niter 1000 --niter_single 0 --niter_step 3 \
    --n_shot 1 --n_frames_G 2 --ref_ratio 0 --display_freq 1 \
    --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid \
    --continue_train
}

train_grid_raw(){
    CUDA_VISIBLE_DEVICES=1,2,3 python train.py --name face8_grid_raw --dataset_mode facefore \
    --adaptive_spade \
    --gpu_ids 0,1,2 --batchSize 18 --nThreads 8 --niter 1000 --niter_single 1001 \
    --n_shot 1 --n_frames_G 1 --ref_ratio 0 \
    --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid \
    --continue_train
}

train_vox_nonlinear(){
    CUDA_VISIBLE_DEVICES=0,3,4,5,6,7 python train.py --name face8_vox_ani_nonlinear --dataset_mode facefore \
    --adaptive_spade --warp_ref --warp_ani --spade_combine --add_raw_loss \
    --gpu_ids 0,1,2,3,4,5 --batchSize 36 --nThreads 64 --niter 1000 --niter_single 1001 \
    --n_shot 8 --n_frames_G 1 \
    --dataroot '/data2/lchen63/voxceleb' --dataset_name vox --save_epoch_freq 50 --display_freq 5000 \
    --continue_train 
}

train_vox_nonlinear_temp(){
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --name face8_vox_ani_nonlinear_temp --dataset_mode facefore \
    --adaptive_spade --warp_ref --warp_ani --spade_combine --add_raw_loss \
    --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 56 --nThreads 64 --niter 1000 --niter_single 6 --niter_step 3 \
    --n_shot 8 --n_frames_G 2 \
    --dataroot '/mnt/Data/lchen63/voxceleb' --dataset_name vox --save_epoch_freq 1 --display_freq 1000 \
    --continue_train --same_flownet
}

train_grid_nonlinear(){
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name face8_grid_ani_nonlinear --dataset_mode facefore \
    --adaptive_spade --warp_ref --spade_combine --add_raw_loss \
    --gpu_ids 0,1,2,3 --batchSize 16 --nThreads 8 --niter 1000 --niter_single 20 --niter_step 3 \
    --n_shot 1 --n_frames_G 2 --ref_ratio 0 \
    --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid --save_epoch_freq 1 \
    --continue_train --same_flownet
}

train_grid_linear_temp(){
    CUDA_VISIBLE_DEVICES=1,2,3 python train.py --name face8_grid_ani_retrain_temp --dataset_mode facefore \
    --adaptive_spade --warp_ref \
    --gpu_ids 0,1,2 --batchSize 12 --nThreads 8 --niter 1000 --niter_single 0 --niter_step 3 \
    --n_shot 1 --n_frames_G 2 --ref_ratio 0 \
    --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid --save_epoch_freq 1 \
    --continue_train --same_flownet
}

train_crema_linear(){
    CUDA_VISIBLE_DEVICES=1,3 python train.py --name face8_crema_linear --dataset_mode facefore \
    --adaptive_spade --warp_ref \
    --gpu_ids 0,1 --batchSize 8 --nThreads 8 --niter 80 --niter_single 81 \
    --n_shot 1 --n_frames_G 1 --ref_ratio 0 \
    --dataroot '/home/cxu-serve/p1/common/CREMA' --dataset_name crema --save_epoch_freq 5 \
    --continue_train
}

train_grid_linear_8shot(){
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name face8_grid_linear --dataset_mode facefore \
    --adaptive_spade --warp_ref \
    --gpu_ids 0,1,2,3 --batchSize 8 --nThreads 8 --niter 30 --niter_single 1001 \
    --n_shot 8 --n_frames_G 1 \
    --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid --save_epoch_freq 5 \
    --continue_train
}

train_vox_nonlinear_comp(){
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --name face8_vox_ani_nonlinear_comp --dataset_mode facefore \
    --adaptive_spade --warp_ref --spade_combine --add_raw_loss \
    --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 64 --nThreads 64 --niter 1000 --niter_single 1001 \
    --n_shot 8 --n_frames_G 1 \
    --dataroot '/mnt/Data/lchen63/voxceleb2' --dataset_name vox --save_epoch_freq 2 --display_freq 1000 \
    --continue_train
}

# train_vox_nonlinear
# train_grid_linear
# train_grid_nonlinear
# train_vox_nonlinear_temp
# train_grid_linear_temp
# train_grid_linear_temp_newflow
# train_grid_raw
# train_crema_linear
# train_grid_linear_8shot
train_vox_nonlinear_comp