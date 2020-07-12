# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch

import pdb

############################# input processing ###################################
def encode_input_finetune(opt, data_list, dummy_bs):
    if opt.isTrain and data_list[0].get_device() == 0:
        data_list = remove_dummy_from_tensor(opt, data_list, dummy_bs)
    tgt_labels, tgt_images, ref_labels, ref_images, warp_ref_lmark, warp_ref_img, ani_lmark, ani_img = data_list

    # target label and image
    tgt_labels = encode_label(opt, tgt_labels)
    tgt_images = tgt_images.cuda() if tgt_images is not None else None

    # flownet ground truth
    # flow_gt = [flow.cuda() if flow is not None else None for flow in flow_gt]
    # conf_gt = [conf.cuda() if conf is not None else None for conf in conf_gt]

    # reference label and image
    ref_labels = encode_label(opt, ref_labels)        
    ref_images = ref_images.cuda()

    # warp reference label and image
    warp_ref_lmark = encode_label(opt, warp_ref_lmark)        
    warp_ref_img = warp_ref_img.cuda()
        
    # for animation
    ani_lmark = encode_label(opt, ani_lmark) if ani_lmark is not None else None
    ani_img = ani_img.cuda() if ani_img is not None else None

    return tgt_labels, tgt_images, ref_labels, ref_images, warp_ref_lmark, warp_ref_img, ani_lmark, ani_img

def encode_input(opt, data_list, dummy_bs):
    if opt.isTrain and data_list[0].get_device() == 0:
        data_list = remove_dummy_from_tensor(opt, data_list, dummy_bs)
    tgt_label, tgt_image, tgt_template, tgt_crop_image, flow_gt, conf_gt, ref_label, ref_image, \
        warp_ref_lmark, warp_ref_img, ori_warping_refs, ani_lmark, ani_img, prev_label, prev_real_image, prev_image = data_list

    # target label and image
    tgt_label = encode_label(opt, tgt_label)
    tgt_image = tgt_image.cuda() if tgt_image is not None else None
    tgt_template = tgt_template.cuda() if tgt_template is not None else None

    # flownet ground truth
    # flow_gt = [flow.cuda() if flow is not None else None for flow in flow_gt]
    # conf_gt = [conf.cuda() if conf is not None else None for conf in conf_gt]

    # reference label and image
    ref_label = encode_label(opt, ref_label)        
    ref_image = ref_image.cuda()

    # animation label and image
    if opt.warp_ani:
        tgt_crop_image = tgt_crop_image.cuda() if tgt_crop_image is not None else None
        ani_lmark = encode_label(opt, ani_lmark)        
        ani_img = ani_img.cuda()

    # warp reference label and image
    warp_ref_lmark = encode_label(opt, warp_ref_lmark)        
    warp_ref_img = warp_ref_img.cuda()
    ori_warping_refs = ori_warping_refs.cuda() if ori_warping_refs is not None else None
        
    return tgt_label, tgt_image, tgt_template, tgt_crop_image, flow_gt, conf_gt, ref_label, ref_image, warp_ref_lmark, \
        warp_ref_img, ori_warping_refs, ani_lmark, ani_img, prev_label, prev_real_image, prev_image

def encode_label(opt, label_map):
    size = label_map.size()
    if len(size) == 5:
        bs, t, c, h, w = size
        label_map = label_map.view(-1, c, h, w)
    else:
        bs, c, h, w = size        

    label_nc = opt.label_nc
    if label_nc == 0:
        input_label = label_map.cuda()
    else:
        # create one-hot vector for label map                         
        label_map = label_map.cuda()
        oneHot_size = (label_map.shape[0], label_nc, h, w)
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.long().cuda(), 1.0)
    
    if len(size) == 5:
        return input_label.view(bs, t, -1, h, w)
    return input_label

def remove_dummy_from_tensor(opt, tensors, remove_size=0):    
    if remove_size == 0: return tensors
    if tensors is None: return None
    if isinstance(tensors, list):
        return [remove_dummy_from_tensor(opt, tensor, remove_size) for tensor in tensors]    
    if isinstance(tensors, torch.Tensor):
        tensors = tensors[remove_size:]
    return tensors