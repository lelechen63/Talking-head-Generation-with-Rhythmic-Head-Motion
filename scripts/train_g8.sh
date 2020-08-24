train_vox_new_nonlinear(){
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name face8_vox_new --dataset_mode facefore \
    --adaptive_spade --warp_ref --warp_ani --spade_combine --add_raw_loss \
    --gpu_ids 0,1,2,3 --batchSize 4 --nThreads 8 --niter 1000 --niter_single 1001 \
    --n_shot 8 --n_frames_G 1 \
    --dataroot '/home/cxu-serve/p1/common/voxceleb2' --dataset_name vox --save_epoch_freq 1 --display_freq 5000 \
    --continue_train --use_new --crop_ref
}

train_vox_origin(){
    CUDA_VISIBLE_DEVICES=3 python train.py --name face8_vox_origin --dataset_mode facefore \
    --adaptive_spade --warp_ref --no_rt \
    --gpu_ids 0 --batchSize 2 --nThreads 4 --niter 1000 --niter_single 1001 \
    --n_shot 8 --n_frames_G 1 \
    --dataroot 'voxceleb2' --dataset_name vox --save_epoch_freq 1 --display_freq 5000 \
    --continue_train
}

train_ouyang_nonlinear(){
    CUDA_VISIBLE_DEVICES=3 python train.py --name face8_ouyang_nonlinear_origin --dataset_mode facefore \
    --adaptive_spade --warp_ref --warp_ani --spade_combine --add_raw_loss \
    --gpu_ids 0 --batchSize 2 --nThreads 4 --niter 5000 --niter_single 5001 \
    --n_shot 8 --n_frames_G 1 \
    --dataroot '/home/cxu-serve/p1/common/demo/oppo_demo' --dataset_name ouyang --save_epoch_freq 15 --display_freq 200 \
    --continue_train --crop_ref
}

train_vox_new_nonlinear
# train_vox_origin
# train_ouyang_nonlinear