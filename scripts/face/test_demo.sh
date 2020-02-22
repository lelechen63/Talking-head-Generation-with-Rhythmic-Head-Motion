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

test_model_grid(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name face8_grid_raw \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --example \
    --n_frames_G 1 \
    --which_epoch $2 \
    --how_many $3 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/grid' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name grid
}

test_model_audio(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_audio.py --name face8_grid_raw \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --example \
    --n_frames_G 1 \
    --which_epoch $2 \
    --how_many $3 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/grid' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name grid
}

# test_model_lrs 3 latest 5   
test_model_vox 2 latest 5
# test_model_audio 2 latest 5
# test_model_grid 2 latest 5