# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import sys
import numpy as np
import torch
from tqdm import tqdm

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from models.loss_collector import loss_backward
from models.trainer import Trainer
from util.distributed import init_dist
from util.distributed import master_only_print as print

import pdb

import warnings
warnings.simplefilter('ignore')

def train():
    opt = TrainOptions().parse()    
    if opt.distributed:
        init_dist()
        opt.batchSize = opt.batchSize // len(opt.gpu_ids)    

    ### setup dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    ### setup trainer    
    trainer = Trainer(opt, data_loader) 

    ### setup models
    model, flowNet = create_model(opt, trainer.start_epoch)
    flow_gt = conf_gt = [None] * 3      
    
    ref_idx_fix = torch.zeros([opt.batchSize])
    for epoch in tqdm(range(trainer.start_epoch, opt.niter + opt.niter_decay + 1)):
        trainer.start_of_epoch(epoch, model, data_loader)
        n_frames_total, n_frames_load = data_loader.dataset.n_frames_total, opt.n_frames_per_gpu
        for idx, data in enumerate(tqdm(dataset), start=trainer.epoch_iter):
            trainer.start_of_iter()            

            if not opt.warp_ani:
                data.update({'ani_image':None, 'ani_lmark':None, 'cropped_images':None, 'cropped_lmarks':None })

            if not opt.no_flow_gt: 
                data_list = [data['tgt_mask_images'], data['cropped_images'], data['warping_ref'], data['ani_image']]
                flow_gt, conf_gt = flowNet(data_list, epoch)
            data_list = [data['tgt_label'], data['tgt_image'], data['tgt_template'], data['cropped_images'], flow_gt, conf_gt]
            data_ref_list = [data['ref_label'], data['ref_image']]
            data_prev = [None, None, None]
            data_ani = [data['warping_ref_lmark'], data['warping_ref'], data['ani_lmark'], data['ani_image']]

            ############## Forward Pass ######################
            prevs = {"raw_images":[], "synthesized_images":[], \
                    "prev_warp_images":[], "prev_weights":[], \
                    "ani_warp_images":[], "ani_weights":[], \
                    "ref_warp_images":[], "ref_weights":[], \
                    "ref_flows":[], "prev_flows":[], "ani_flows":[], \
                    "ani_syn":[]}
            for t in range(0, n_frames_total, n_frames_load):
                
                data_list_t = get_data_t(data_list, n_frames_load, t) + data_ref_list + \
                              get_data_t(data_ani, n_frames_load, t) + data_prev

                # get new previous flow loss
                # if t != 0:
                #     with torch.no_grad():
                #         flow_prev_gt, conf_prev_gt = flowNet.module.flowNet_forward(data_prev[2].cuda(), data_list_t[1].cuda())
                #         data_list_t[4][1] = flow_prev_gt
                #         data_list_t[5][1] = conf_prev_gt

                g_losses, generated, data_prev, ref_idx = model(data_list_t, save_images=trainer.save, mode='generator', ref_idx_fix=ref_idx_fix)
                g_losses = loss_backward(opt, g_losses, model.module.optimizer_G)

                d_losses, _ = model(data_list_t, mode='discriminator', ref_idx_fix=ref_idx_fix)
                d_losses = loss_backward(opt, d_losses, model.module.optimizer_D)

                # store previous
                store_prev(generated, prevs)
                        
            loss_dict = dict(zip(model.module.lossCollector.loss_names, g_losses + d_losses))     

            # output_data_list = generated + data_list + [data['ref_label'], data['ref_image']] + data_ani + [data['cropped_lmarks']]
            output_data_list = [prevs] + [data['ref_image']] + data_ani + data_list + [data['tgt_mask_images']]

            if trainer.end_of_iter(loss_dict, output_data_list, model):
                break        

        trainer.end_of_epoch(model)

def get_data_t(data, n_frames_load, t):
    if data is None: return None
    if type(data) == list:
        return [get_data_t(d, n_frames_load, t) for d in data]
    return data[:,t:t+n_frames_load]

def store_prev(data, prevs):
    fake_image, fake_raw_image, img_ani, warped_image, flow, weight, atn_score = data

    prevs['raw_images'].append(fake_raw_image[-1].cpu().data if fake_raw_image is not None else None)
    prevs['synthesized_images'].append(fake_image[-1].cpu().data)
    prevs['prev_warp_images'].append(warped_image[1][-1].cpu().data if warped_image[1] is not None else None)
    prevs['prev_weights'].append(weight[1][-1].cpu().data if weight[1] is not None else None)
    prevs['ani_warp_images'].append(warped_image[2][-1].cpu().data if warped_image[2] is not None else None)
    prevs['ani_weights'].append(weight[2][-1].cpu().data if weight[2] is not None else None)
    prevs['ref_warp_images'].append(warped_image[0][-1].cpu().data if warped_image[0] is not None else None)
    prevs['ref_weights'].append(weight[0][-1].cpu().data if weight[0] is not None else None)
    prevs['ref_flows'].append(flow[0][-1].cpu().data if flow[0] is not None else None)
    prevs['prev_flows'].append(flow[1][-1].cpu().data if flow[1] is not None else None)
    prevs['ani_flows'].append(flow[2][-1].cpu().data if flow[2] is not None else None)
    prevs['ani_syn'].append(img_ani[-1].cpu().data if img_ani is not None else None)

if __name__ == "__main__":
   train()