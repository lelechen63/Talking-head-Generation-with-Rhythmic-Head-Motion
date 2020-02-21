import torch
import torch.nn as nn
import numpy as np

import models.networks as networks
from models.base_model import BaseModel
from models.input_process import *
from models.loss_collector import LossCollector, loss_backward
from models.face_refiner import FaceRefineModel

class TempModel(nn.Module):
    def name(self):
        return 'Temp_Package_Vid2VidModel'

    def initialize(self, flowNet, v2vModel):
        self.flowNet = flowNet
        self.v2vModel = v2vModel

    def forward(self, opt, data, epoch, n_frames_total, n_frames_load, save_images):
        # ground truth flownet
        if not opt.no_flow_gt: 
            data_list = [data['tgt_image'], data['cropped_images'], data['warping_ref'], data['ani_image']]
            flow_gt, conf_gt = self.flowNet(data_list, epoch)
        else:
            flow_gt, conf_gt = None, None
        data_list = [data['tgt_label'], data['tgt_image'], data['cropped_images'], flow_gt, conf_gt]
        data_ref_list = [data['ref_label'], data['ref_image']]
        data_prev = [None, None, None]
        data_ani = [data['warping_ref_lmark'], data['warping_ref'], data['ani_lmark'], data['ani_image']]

        ############## Forward Pass ######################
        for t in range(0, n_frames_total, n_frames_load):
            
            data_list_t = self.get_data_t(data_list, n_frames_load, t) + data_ref_list + \
                            self.get_data_t(data_ani, n_frames_load, t) + data_prev
                            
            g_losses, generated, data_prev, ref_idx = self.v2vModel(data_list_t, save_images=save_images, mode='generator', ref_idx_fix=None)
            g_losses = loss_backward(opt, g_losses, self.v2vModel.module.optimizer_G)

            d_losses, _ = self.v2vModel(data_list_t, mode='discriminator', ref_idx_fix=None)
            d_losses = loss_backward(opt, d_losses, self.v2vModel.module.optimizer_D)

        return generated, flow_gt, conf_gt

    # ground truth flownet
    def forward_flowNet(self, opt, data, epoch):
        if not opt.no_flow_gt: 
            data_list = [data['tgt_image'], data['cropped_images'], data['warping_ref'], data['ani_image']]
            flow_gt, conf_gt = self.flowNet(data_list, epoch)
        else:
            flow_gt, conf_gt = None, None

        return flow_gt, conf_gt

    def forward_gen(self, opt, data, epoch, n_frames_total, n_frames_load, save_images):
        flow_gt, conf_gt = self.forward_flowNet(opt, data, epoch)

        data_list = [data['tgt_label'], data['tgt_image'], data['cropped_images'], flow_gt, conf_gt]
        data_ref_list = [data['ref_label'], data['ref_image']]
        data_prev = [None, None, None]
        data_ani = [data['warping_ref_lmark'], data['warping_ref'], data['ani_lmark'], data['ani_image']]

        ############## Forward Pass ######################
        for t in range(0, n_frames_total, n_frames_load):
            
            data_list_t = self.get_data_t(data_list, n_frames_load, t) + data_ref_list + \
                            self.get_data_t(data_ani, n_frames_load, t) + data_prev
                            
            g_losses, generated, data_prev, ref_idx = self.v2vModel(data_list_t, save_images=save_images, mode='generator', ref_idx_fix=None)
            g_losses = loss_backward(opt, g_losses, self.v2vModel.module.optimizer_G)

            d_losses, _ = self.v2vModel(data_list_t, mode='discriminator', ref_idx_fix=None)
            d_losses = loss_backward(opt, d_losses, self.v2vModel.module.optimizer_D)

        return generated, flow_gt, conf_gt


    def get_data_t(self, data, n_frames_load, t):
        if data is None: return None
        if type(data) == list:
            return [self.get_data_t(d, n_frames_load, t) for d in data]
        return data[:,t:t+n_frames_load]