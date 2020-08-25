train_vox_new_nonlinear(){
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --name face8_vox_new \
    --dataset_mode facefore \
    --adaptive_spade \
    --warp_ref \
    --warp_ani \
    --spade_combine \
    --add_raw_loss \
    --gpu_ids 0,1,2,3 \
    --batchSize 4 \
    --nThreads 8 \
    --niter 1000 \
    --niter_single 1001 \
    --n_shot 8 \
    --n_frames_G 1 \
    --dataroot 'voxceleb2' \
    --dataset_name vox \
    --save_epoch_freq 1 \
    --display_freq 5000 \
    --continue_train \
    --use_new \
    --crop_ref
}

train_vox_new_nonlinear