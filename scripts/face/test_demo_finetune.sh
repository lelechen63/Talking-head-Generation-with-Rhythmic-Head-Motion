test_model_crema_finetune(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_finetune.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --spade_combine \
    --add_raw_loss \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/CREMA' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name crema \
    --finetune_shot 32 \
    --finetune \
    --crop_ref \
    --find_largest_mouth
}

test_model_crema_finetune_save(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_finetune_save.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --spade_combine \
    --add_raw_loss \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/CREMA' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name crema \
    --finetune_shot 32 \
    --finetune \
    --crop_ref \
    --find_largest_mouth
}

test_model_grid_finetune(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_finetune.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/grid' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name grid \
    --finetune_shot 32 \
    --finetune \
    --crop_ref \
    --find_largest_mouth
}

test_model_obama(){
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
    --dataroot '/home/cxu-serve/p1/common/Obama' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name obama \
    --finetune_shot 64 \
    --crop_ref \
    --finetune \
    --origin_not_require
}

test_model_obama_8(){
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
    --dataroot '/home/cxu-serve/p1/common/Obama' \
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8 \
    --serial_batches \
    --dataset_name obama \
    --finetune_shot 64 \
    --crop_ref \
    --finetune \
    --origin_not_require
}

test_model_obama_32(){
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
    --dataroot '/home/cxu-serve/p1/common/Obama' \
    --ref_img_id "0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155" \
    --n_shot 32 \
    --serial_batches \
    --dataset_name obama \
    --finetune_shot 64 \
    --crop_ref \
    --finetune \
    --origin_not_require
}

test_model_grid_linear(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_finetune.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/grid' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name grid \
    --finetune_shot 32 \
    --finetune \
    --crop_ref \
    --find_larges
}

test_model_obama_8_front(){
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
    --dataroot '/home/cxu-serve/p1/common/Obama' \
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8 \
    --serial_batches \
    --dataset_name obama_front \
    --finetune_shot 64 \
    --crop_ref \
    --finetune \
    --origin_not_require \
    --no_head_motion
}

test_model_obama_8_fake(){
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
    --dataroot '/home/cxu-serve/p1/common/Obama' \
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8 \
    --serial_batches \
    --dataset_name obama_fake \
    --finetune_shot 64 \
    --crop_ref \
    --finetune \
    --origin_not_require
}

test_model_other_32_fake(){
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
    --dataroot '/home/cxu-serve/p1/common/voxceleb2' \
    --ref_img_id "0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93" \
    --n_shot 32 \
    --serial_batches \
    --dataset_name other_fake \
    --finetune_shot 64 \
    --crop_ref \
    --finetune \
    --origin_not_require
}

test_model_other_32_fake_front(){
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
    --dataroot '/home/cxu-serve/p1/common/voxceleb2' \
    --ref_img_id "0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93" \
    --n_shot 32 \
    --serial_batches \
    --dataset_name other_fake \
    --finetune_shot 64 \
    --crop_ref \
    --finetune \
    --origin_not_require \
    --no_head_motion
}

test_model_obama_32_fake(){
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
    --dataroot '/home/cxu-serve/p1/common/Obama' \
    --ref_img_id "0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155" \
    --n_shot 128 \
    --serial_batches \
    --dataset_name obama_fake \
    --finetune_shot 64 \
    --crop_ref \
    --finetune \
    --origin_not_require
}

test_model_lrs(){
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
    --dataroot '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name lrs \
    --finetune_shot 1 \
    --crop_ref \
    --finetune \
    --origin_not_require
}

test_model_lrs_32(){
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
    --dataroot '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4' \
    --ref_img_id "0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62" \
    --n_shot 32 \
    --serial_batches \
    --dataset_name lrs \
    --finetune_shot 32 \
    --crop_ref \
    --finetune \
    --origin_not_require
}



# test_model_crema_finetune 3 face8_crema_mouth_nonlinear_L1 40 50
# test_model_crema_finetune 3 face8_crema_mouth_nonlinear latest 50
# test_model_crema_finetune_save 1 face8_crema_mouth_nonlinear latest 50
# test_model_grid_finetune 2 face8_grid_linear_mask latest 50
# test_model_grid_finetune 2 face8_grid_linear_mask finetune 50
# test_model_grid_linear 2 face8_grid_linear latest 50
# test_model_obama 3 face8_vox_ani_nonlinear_continue 21 50
test_model_obama_8 3 face8_obama_nonlinear 1400 50
# test_model_obama_8_front 3 face8_obama_nonlinear 1200 50
# test_model_obama_8_fake 3 face8_obama_nonlinear 50 50
# test_model_obama_32_fake 3 face8_obama_nonlinear 1400 50
# test_model_other_32_fake 3 face8_vox_ani_nonlinear_continue 21 50
# test_model_other_32_fake_front 2 face8_vox_ani_nonlinear_continue 21 50
# test_model_obama_32 3 face8_obama_nonlinear 1400 50
# test_model_lrs 2 face8_vox_ani_nonlinear_continue 21 50
# test_model_lrs 2 face8_lrs_nonlinear 20 50
# test_model_lrs_32 2 face8_vox_ani_nonlinear_continue 21 50