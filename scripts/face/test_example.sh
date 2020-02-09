# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt

test_model(){
    CUDA_VISIBLE_DEVICES=$1 python test_example.py --name face8_linear \
    --dataset_mode fewshot_face_pickle \
    --adaptive_spade \
    --warp_ref \
    --example \
    --which_epoch $2 \
    --how_many $3 \
    --nThreads 0 \
    --dataroot '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/' \
    --ref_img_id "0,10,20,30,40,50,60,70" \
    --n_shot 8
}

test_model 0 50 400