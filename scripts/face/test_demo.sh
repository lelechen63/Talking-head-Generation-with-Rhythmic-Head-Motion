test_model_lrs(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name face8_vox_ani_nonlinear \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --warp_ani \
    --add_raw_loss \
    --spade_combine \
    --example \
    --n_frames_G 1 \
    --which_epoch $2 \
    --how_many $3 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4' \
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8 \
    --serial_batches \
    --dataset_name lrs
}

test_model_vox(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name face8_vox_ani_nonlinear \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --warp_ani \
    --add_raw_loss \
    --spade_combine \
    --example \
    --n_frames_G 1 \
    --which_epoch $2 \
    --how_many $3 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/voxceleb2' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name vox
}

test_model_vox_temp(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
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
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8 \
    --serial_batches \
    --dataset_name vox
}

test_model_vox_new(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
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
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name vox \
    --crop_ref
}

test_model_grid(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/grid' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name grid \
    --find_largest_mouth
}

test_model_audio(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_audio.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/grid' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name grid
}

test_grid_save(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_save.py --name $2 \
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
    --batchSize 5 \
    --serial_batches \
    --dataset_name grid \
    --find_largest_mouth
}

test_model_crema(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/CREMA' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name crema
}

test_model_lisa(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa \
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
    --dataroot '/home/cxu-serve/p1/common/demo' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name lisa
}

test_model_lrw(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
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
    --dataroot '/home/cxu-serve/p1/common/lrw' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name lrw \
    --crop_ref
}

test_model_crema_multi(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_crema.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/CREMA' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name crema
}

test_lrw_save(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_save.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --warp_ani \
    --spade_combine \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/lrw' \
    --ref_img_id "0" \
    --n_shot 1 \
    --batchSize 5 \
    --serial_batches \
    --dataset_name lrw \
    --find_largest_mouth
}

# test_model_lrs 3 latest 5
# test_model_vox 3 latest 5
# test_model_vox_temp 2 face8_vox_ani_nonlinear_temp latest 5
# test_model_vox_temp 1 face8_vox_ani_nonlinear_atten latest 5
# test_model_vox_temp 1 face8_vox_ani_retrain latest 5
# test_model_vox_temp 1 face8_vox_ani_nonlinear_continue 9 5
# test_model_audio 2 latest 5
# test_model_grid 1 face8_grid_ani_retrain latest 50
# test_grid_save 2 face8_grid_ani_retrain latest 100
# test_model_crema 3 face8_crema_linear 50 20
# test_model_lisa 3 face8_vox_ani_nonlinear_atten latest 3
# test_model_lrw 1 face8_vox_ani_nonlinear_continue 9 100
# test_model_crema_multi 3 face8_crema_linear 50 50
# test_lrw_save 1 face8_vox_ani_nonlinear_atten latest 100
# test_model_grid 2 face8_grid_linear_mask 8 5
# test_model_audio 1 face8_grid_linear_mask latest 20
test_model_vox_new 1 face8_vox_ani_nonlinear_continue 11 10