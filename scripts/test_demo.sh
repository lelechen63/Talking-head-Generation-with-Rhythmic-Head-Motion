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
    --nThreads 0 \
    --dataroot 'demo' \
    --ref_img_id "0" \
    --n_shot 8 \
    --serial_batches \
    --dataset_name vox \
    --crop_ref \
    --use_new
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
    --nThreads 0 \
    --dataroot 'demo' \
    --ref_img_id "0" \
    --n_shot 8 \
    --serial_batches \
    --dataset_name vox \
    --crop_ref \
    --use_new \
    --origin_not_require \
    --finetune \
    --finetune_shot 8
}

test_model_vox_ani 3 face8_vox_new latest 50
# test_model_vox_ani_finetune 3 face8_vox_new latest 1