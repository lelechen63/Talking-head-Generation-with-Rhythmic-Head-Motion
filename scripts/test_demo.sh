test_model_vox_ani(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_ani.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --warp_ani \
    --add_raw_loss \
    --spade_combine \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 4 \
    --dataroot 'demo' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name vox \
    --crop_ref
}

test_model_vox_ani_finetune(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_finetune.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --warp_ani \
    --add_raw_loss \
    --spade_combine \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 4 \
    --dataroot 'demo' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name vox \
    --crop_ref \
    --finetune \
    --finetune_shot 1
}

# test_model_vox_ani 3 face8_vox_ani_nonlinear_continue 21 10
test_model_vox_ani_finetune 3 face8_vox_ani_nonlinear_continue 21 10