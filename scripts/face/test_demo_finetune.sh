test_model_crema_finetune(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_finetune.py --name $2 \
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
    --finetune_shot 2 \
    --finetune
}

test_model_crema_finetune 1 face8_crema_linear 20 50