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
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name face8_grid_ani --dataset_mode facefore \
    --adaptive_spade --warp_ref \
    --gpu_ids 0,1,2,3 --batchSize 12 --nThreads 8 --niter 1000 --niter_single 1001 \
    --n_shot 8 --n_frames_G 1 \
    --dataroot '/home/cxu-serve/p1/common/grid' --dataset_name grid \
    --continue_train
}

train_vox_nonlinear(){
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --name face8_vox_ani_nonlinear --dataset_mode facefore \                                                      
    --adaptive_spade --warp_ref --warp_ani --spade_combine \                                                                                                                   
    --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 72 --nThreads 64 --niter 1000 --niter_single 1001 \                                                                   
    --n_shot 8 --n_frames_G 1 \                                                                                                                             
    --dataroot '/mnt/Data/lchen63/voxceleb2' --dataset_name vox --save_epoch_freq 50 --display_freq 5000 \                                                    
    --continue_train 
}