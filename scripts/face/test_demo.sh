test_model_lrs_32(){
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
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4' \
    --ref_img_id "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62" \
    --n_shot 32 \
    --serial_batches \
    --dataset_name lrs \
    --crop_ref
}

test_model_lrs(){
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
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name lrs \
    --crop_ref
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
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8 \
    --serial_batches \
    --dataset_name vox \
    --crop_ref
}

test_model_vox_new_noani_2(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
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

test_model_vox_audio(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
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
    --crop_ref \
    --audio_drive
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

test_model_crema_nonlinear(){
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
    --dataset_name crema \
    --crop_ref
}

test_model_lisa(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name lisa \
    --crop_ref \
    --finetune
}

test_model_vincent(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name vincent \
    --crop_ref \
    --finetune
}

test_model_groot(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name groot \
    --crop_ref \
    --finetune
}

test_model_hulk(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name hulk \
    --crop_ref \
    --finetune
}

test_model_superman(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name superman \
    --crop_ref \
    --finetune
}

test_model_picasso(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name picasso \
    --crop_ref \
    --finetune
}

test_model_turing(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name turing \
    --crop_ref \
    --finetune
}

test_model_frid(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name frid \
    --crop_ref \
    --finetune
}

test_model_self1(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name self1 \
    --crop_ref \
    --finetune
}

test_model_self2(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name self2 \
    --crop_ref \
    --finetune
}

test_model_mulan(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name mulan \
    --crop_ref \
    --finetune
}

test_model_david(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_lisa.py --name $2 \
    --dataset_mode facefore_demo_lisa2 \
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
    --dataset_name david \
    --crop_ref \
    --finetune
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

test_model_face(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $3 \
    --how_many $4 \
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/voxceleb2' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name vox
}

test_model_vox_new_32(){
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
    --nThreads 8 \
    --dataroot '/home/cxu-serve/p1/common/voxceleb2' \
    --ref_img_id "0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93" \
    --n_shot 32 \
    --serial_batches \
    --dataset_name vox \
    --crop_ref
}

test_model_vox_new_linear(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --warp_ani \
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

test_model_vox_new_nowarp(){
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
    --crop_ref \
    --no_warp
}

test_model_vox_new_noani(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name $2 \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
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

test_model_vox_new_noatten(){
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
    --dataset_name vox \
    --crop_ref \
    --no_atten
}

test_model_vox_new_noheadmotion(){
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
    --crop_ref \
    --no_head_motion
}

# test_model_david 3 face8_vox_ani_nonlinear_continue 21 200
# test_model_mulan 3 face8_vox_ani_nonlinear_continue 21 200
# test_model_vincent 1 face8_vox_ani_nonlinear_continue 21 200
# test_model_vincent 1 face8_lrs_nonlinear 21 200
# test_model_vincent 1 face8_lrs_nonlinear 50 200
# test_model_lrs 1 face8_vox_ani_nonlinear_continue 21 200
# test_model_lrw 1 face8_vox_ani_nonlinear_continue 21 50
# test_model_lrs 3 latest 5
# test_model_vox 3 latest 5
# test_model_vox_temp 2 face8_vox_ani_nonlinear_temp latest 5
# test_model_vox_temp 1 face8_vox_ani_nonlinear_atten latest 5
# test_model_vox_temp 1 face8_vox_ani_retrain latest 5
# test_model_vox_temp 1 face8_vox_ani_nonlinear_continue 9 5 
# test_model_audio 2 latest 5
# test_model_grid 1 face8_grid_ani_retrain latest 50
# test_grid_save 2 face8_grid_ani_retrain latest 100
# test_model_crema 3 face8_crema_linear 20 20000
# test_model_crema_nonlinear 3 face8_crema_nonlinear 20 3
# test_model_lisa 3 face8_vox_ani_nonlinear_continue 21 3
# test_model_lisa 3 face8_lrs_nonlinear 30 3
# test_model_groot 2 face8_vox_ani_nonlinear_continue 21 3
# test_model_hulk 3 face8_vox_ani_nonlinear_continue 21 3
# test_model_hulk 3 face8_lrs_nonlinear 50 3
# test_model_hulk 3 face8_lrs_nonlinear_full 30 3
# test_model_picasso 3 face8_vox_ani_nonlinear_continue 21 3
# test_model_picasso 3 face8_lrs_nonlinear 40 3
# test_model_picasso 3 face8_lrs_nonlinear_full 30 3
# test_model_turing 3 face8_vox_ani_nonlinear_continue 21 3
# test_model_frid 3 face8_vox_ani_nonlinear_continue 21 3
# test_model_frid 3 face8_lrs_nonlinear 40 3
# test_model_self1 3 face8_vox_ani_nonlinear_continue 21 3
# test_model_self1 3 face8_lrs_nonlinear 20 3
# test_model_self2 3 face8_vox_ani_nonlinear_continue 21 3
# test_model_self2 3 face8_lrs_nonlinear 20 3
# test_model_superman 2 face8_vox_ani_nonlinear_continue 21 3
# test_model_superman 2 face8_lrs_nonlinear_full 40 3
# test_model_vincent 2 face8_vox_ani_nonlinear_continue 21 3
# test_model_lrw 1 face8_vox_ani_nonlinear_continue 9 100
# test_model_crema_multi 3 face8_crema_linear 50 50
# test_lrw_save 1 face8_vox_ani_nonlinear_atten latest 100
# test_model_grid 2 face8_grid_linear_mask 8 5
# test_model_audio 1 face8_grid_linear_mask latest 20
# test_model_vox_new 0 face8_vox_ani_nonlinear_continue 21 2
test_model_vox_new_noani_2 1 face8_vox_ani_nonlinear_noani 5 500
# test_model_vox_audio 0 face8_vox_audio_nonlinear latest 2
# test_model_face 1 face8_previous 105 3
# test_model_vox_new_32 1 face8_vox_ani_nonlinear_continue 21 200
# test_model_vox_new_nowarp 3 no_warp latest 5
# test_model_vox_new_noatten 1 no_atten latest 5
# test_model_vox_new_noani 3 no_ani latest 5
# test_model_vox_new 1 no_ani latest 200
# test_model_vox_new 1 no_atten latest 200
# test_model_vox_new_linear 1 face8_vox_ani latest 5
# test_model_vox_new_noheadmotion 1 face8_vox_ani_nonlinear_continue 21 200