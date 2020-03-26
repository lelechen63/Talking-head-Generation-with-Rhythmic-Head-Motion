test_model_vox(){
    CUDA_VISIBLE_DEVICES=$1 python test_demo_example.py --name face8_vox_ani_nonlinear_continue \
    --dataset_mode facefore_demo \
    --adaptive_spade \
    --warp_ref \
    --warp_ani \
    --add_raw_loss \
    --spade_combine \
    --example \
    --n_frames_G 1 \
    --which_epoch $2 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/voxceleb2' \
    --ref_dataroot '/home/cxu-serve/p1/common/voxceleb2' \
    --ref_img_id "0" \
    --n_shot 1 \
    --serial_batches \
    --dataset_name vox \
    --tgt_video_path "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/utfjXffHDgg/00198_aligned.mp4" \
    --ref_dataset vox \
    --ref_video_path "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/utfjXffHDgg/00198_aligned.mp4" \
    --ref_ani_id 106 \
    --finetune \
    --origin_not_require \
    --crop_ref
}

test_model_vox 0 21