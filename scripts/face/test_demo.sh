test_model(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo.py --name face8_grid_ani \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --example \
    --n_frames_G 1 \
    --which_epoch $2 \
    --how_many $3 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/grid' \
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8 \
    --serial_batches
    --dataset_name grid
}

test_model 3 latest 5