# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt

test_model(){
    CUDA_VISIBLE_DEVICES=$1 python test.py --name face8_linear \
    --dataset_mode fewshot_face \
    --adaptive_spade \
    --warp_ref \
    --which_epoch $2 \
    --how_many $3 \
    --nThreads 0 \
    --seq_path "/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped/images/"$4 \
    --ref_img_path "/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped/images/"$5 \
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8
}

frame2video(){
    python f2v.py --root "/home/cxu-serve/u1/gcui2/code/audioFace/few-shot-vid2vid-new/results/face8_linear/test_50/images-0,10,20,30,40,50,60,70_images"
}

test_model_pickle(){
    CUDA_VISIBLE_DEVICES=$1 python test_example.py --name face8_linear \
    --dataset_mode fewshot_face_pickle \
    --adaptive_spade \
    --warp_ref \
    --which_epoch $2 \
    --how_many $3 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/' \
    --seq_path "/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped/videos/"$4".mp4" \
    --ref_img_path "/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped/videos/"$5".mp4" \
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8
}

# test_model 3 35 450 799 799
# frame2video

test_model_pickle 3 50 450 800 800

# test_model 3 50 450 800 800
# frame2video
# test_model 3 50 450 801 801
# frame2video
# test_model 3 50 450 802 802
# frame2video
# test_model 3 50 450 803 803
# frame2video
# test_model 3 50 450 804 804
# frame2video
# test_model 3 50 450 805 805
# frame2video