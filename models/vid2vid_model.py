# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import models.networks as networks
from models.base_model import BaseModel
from models.input_process import *
from models.loss_collector import LossCollector, loss_backward
from models.face_refiner import FaceRefineModel

import pdb

class Vid2VidModel(BaseModel):
    def name(self):
        return 'Vid2VidModel'

    def initialize(self, opt, epoch=0):
        BaseModel.initialize(self, opt)        
        torch.backends.cudnn.benchmark = True        
        
        # define losses        
        self.lossCollector = LossCollector()
        self.lossCollector.initialize(opt)        
        
        # define networks
        self.define_networks(epoch)

        # load networks        
        self.load_networks()      

    def forward(self, data_list, save_images=False, mode='inference', dummy_bs=0, ref_idx_fix=None, epoch=0):            
        tgt_label, tgt_image, tgt_template, tgt_crop_image, flow_gt, conf_gt, ref_labels, ref_images, \
            warp_ref_lmark, warp_ref_img, ori_warping_refs, ani_lmark, ani_img, prev_label, prev_real_image, prev_image, \
        = encode_input(self.opt, data_list, dummy_bs)
        
        if mode == 'generator':            
            g_loss, generated, prev, ref_idx = self.forward_generator(tgt_label, tgt_image, tgt_template, tgt_crop_image, flow_gt, conf_gt, ref_labels, ref_images,
                warp_ref_lmark, warp_ref_img, ori_warping_refs, ani_lmark, ani_img,
                prev_label, prev_real_image, prev_image, ref_idx_fix, epoch)
            # return g_loss, generated if save_images else [], prev, ref_idx
            return g_loss, generated, prev, ref_idx

        elif mode == 'discriminator':            
            d_loss, ref_idx = self.forward_discriminator(tgt_label, tgt_image, tgt_template, ref_labels, ref_images, 
                warp_ref_lmark, warp_ref_img, ori_warping_refs, ani_lmark, ani_img, 
                prev_label, prev_real_image, prev_image, ref_idx_fix)
            return d_loss, ref_idx

        else:
            return self.inference(tgt_label, ref_labels, ref_images, warp_ref_lmark, warp_ref_img, ani_lmark, ani_img, ref_idx_fix)
   
    def forward_generator(self, tgt_label, tgt_image, tgt_template, tgt_crop_image, flow_gt, conf_gt, ref_labels, ref_images, 
                          warp_ref_lmark=None, warp_ref_img=None, ori_warping_refs=None, ani_lmark=None, ani_img=None, 
                          prev_label=None, prev_real_image=None, prev_image=None, ref_idx_fix=None, epoch=0):

        ### fake generation
        [fake_image, fake_raw_image, img_ani, warped_image, flow, weight], \
            [ref_label, ref_image], [prev_label, prev_real_image, prev_image], atn_score, ref_idx = \
            self.generate_images(tgt_label, tgt_image, ref_labels, ref_images, warp_ref_lmark, warp_ref_img, ani_lmark, ani_img, [prev_label, prev_real_image, prev_image], ref_idx_fix)

        ### temporal losses
        nets = self.netD, self.netDT
        loss_GT_GAN, loss_GT_GAN_Feat = self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)
        if self.isTrain and prev_label is not None:
            tgt_image_all = torch.cat([prev_real_image, tgt_image], dim=1)
            fake_image_all = torch.cat([prev_image, fake_image], dim=1)
            data_list = [None, tgt_image_all, fake_image_all, None, None]
            loss_GT_GAN, loss_GT_GAN_Feat = self.lossCollector.compute_GAN_losses(nets, data_list, 
                for_discriminator=False, for_temporal=True)

        ### individual frame losses
        # GAN loss
        loss_G_GAN, loss_G_GAN_Feat = self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)
        loss_ssmi = self.Tensor(1).fill_(0)
        if img_ani is not None:
            data_list = [tgt_label, [tgt_image, tgt_image, tgt_image], [fake_image, fake_raw_image, img_ani], ref_label, ref_image]
        else:
            data_list = [tgt_label, [tgt_image, tgt_image], [fake_image, fake_raw_image], ref_label, ref_image]
        loss_G_GAN, loss_G_GAN_Feat = self.lossCollector.compute_GAN_losses(nets,
            data_list, for_discriminator=False)
            
        # for mouth discriminator
        loss_GM_GAN, loss_GM_GAN_Feat = self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)
        loss_M_ssmi = self.Tensor(1).fill_(0)

        # VGG loss
        loss_G_VGG = self.lossCollector.compute_VGG_losses(fake_image, fake_raw_image, img_ani, tgt_image)
        loss_GM_VGG = self.Tensor(1).fill_(0)

        # L1 loss
        loss_l1 = self.lossCollector.compute_L1_loss(syn_image=fake_image, tgt_image=tgt_image)
        loss_atten = self.lossCollector.atten_L1_loss(syn_image=fake_image, tgt_image=tgt_image, tgt_template=tgt_template)

        flow, weight, flow_gt, conf_gt, warped_image, tgt_image, tgt_crop_image = \
            self.reshape([flow, weight, flow_gt, conf_gt, warped_image, tgt_image, tgt_crop_image])             
        loss_F_Flow, loss_F_Warp = self.lossCollector.compute_flow_losses(flow, warped_image, tgt_image, tgt_crop_image,
            flow_gt, conf_gt)

        # add W
        loss_W = self.lossCollector.compute_weight_losses(weight, warped_image, tgt_image, tgt_crop_image)
        
        loss_list = [loss_G_GAN, loss_G_GAN_Feat, 
                    loss_G_VGG, loss_GM_VGG, # GAN + VGG loss
                    loss_GT_GAN, loss_GT_GAN_Feat,           # temporal GAN loss
                    loss_GM_GAN, loss_GM_GAN_Feat,           # Mouth GAN loss
                    loss_ssmi, loss_M_ssmi,         # ssmi loss
                    loss_F_Flow, loss_F_Warp, loss_l1, loss_atten, loss_W]        # flow loss (debug whether add weight loss)
                
        loss_list = [loss.view(1, 1) for loss in loss_list]

        return loss_list, \
               [fake_image, fake_raw_image, img_ani, warped_image, flow, weight, atn_score], \
               [prev_label, prev_real_image, prev_image], \
               ref_idx
    
    def forward_discriminator(self, tgt_label, tgt_image, tgt_template, ref_labels, ref_images, 
                              warp_ref_lmark=None, warp_ref_img=None, ori_warping_refs=None, ani_lmark=None, ani_img=None, 
                              prev_label=None, prev_real_image=None, prev_image=None, ref_idx_fix=None):
        ### Fake Generation
        with torch.no_grad():
            [fake_image, fake_raw_image, img_ani, _, _, _], [ref_label, ref_image], _, _, ref_idx = \
                self.generate_images(tgt_label, tgt_image, ref_labels, ref_images, warp_ref_lmark, \
                    warp_ref_img, ani_lmark, ani_img, [prev_label, prev_real_image, prev_image], ref_idx_fix)

        ### temporal losses
        nets = self.netD, self.netDT
        loss_temp = []
        if self.isTrain:
            if prev_image is None: prev_image = tgt_image.repeat(1, self.opt.n_frames_G-1, 1, 1, 1)
            tgt_image_all = torch.cat([prev_image, tgt_image], dim=1)
            fake_image_all = torch.cat([prev_image, fake_image], dim=1)            
            data_list = [None, tgt_image_all, fake_image_all, None, None]
            loss_temp = self.lossCollector.compute_GAN_losses(nets, data_list, for_discriminator=True, for_temporal=True)

        ### individual frame losses
        if img_ani is not None:
            data_list = [tgt_label, [tgt_image, tgt_image, tgt_image], [fake_image, fake_raw_image, img_ani], ref_label, ref_image]
        else:
            data_list = [tgt_label, [tgt_image, tgt_image], [fake_image, fake_raw_image], ref_label, ref_image]
        loss_indv = self.lossCollector.compute_GAN_losses(nets, data_list, for_discriminator=True)

        # mouth loss
        loss_M = [self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)]

        loss_list = list(loss_indv) + list(loss_temp) + list(loss_M)
        loss_list = [loss.view(1, 1) for loss in loss_list]
        return loss_list, ref_idx              

    def generate_images(self, tgt_labels, tgt_images, ref_labels, ref_images, 
                        warp_ref_lmark=None, warp_ref_img=None, ani_lmark=None, ani_img=None,
                        prevs=[None, None, None], ref_idx_fix=None):
        opt = self.opt      
        generated_images, atn_score = [None] * 6, None 
        ref_labels_valid = ref_labels
        
        for t in range(opt.n_frames_per_gpu):
            # get inputs for time t
            tgt_label_t, tgt_label_valid, tgt_images, warp_ref_img_t, warp_ref_lmark_t, \
                    ani_img_t, ani_lmark_t, prev_t = self.get_input_t(tgt_labels, tgt_images, warp_ref_img, 
                                                                    warp_ref_lmark, ani_img, 
                                                                    ani_lmark, prevs, t)

            # actual network forward
            fake_image, flow, weight, fake_raw_image, warped_image, atn_score, ref_idx, img_ani \
                = self.netG(tgt_label_valid, ref_labels_valid, ref_images, prev_t, \
                            warp_ref_img_t, warp_ref_lmark_t, ani_img_t, ani_lmark_t, \
                            ref_idx_fix=ref_idx_fix)
            
            # ref_label_valid, ref_label_t, ref_image_t = self.netG.pick_ref([ref_labels_valid, ref_labels, ref_images], ref_idx)
            ref_label_valid, ref_image_t = warp_ref_lmark_t, warp_ref_img_t
                        
            # concatenate current output with previous outputs
            generated_images = self.concat([generated_images, [fake_image, fake_raw_image, img_ani, warped_image, flow, weight]], dim=1)
            prevs = self.concat_prev(prevs, [tgt_label_valid, tgt_images, fake_image])

        return generated_images, [ref_label_valid, ref_image_t], prevs, atn_score, ref_idx

    def get_input_t(self, tgt_labels, tgt_images, warp_ref_img, warp_ref_lmark, ani_img, ani_lmark, prevs, t):
        b, _, _, h, w = tgt_labels.shape        
        tgt_label = tgt_labels[:,t]
        tgt_image = tgt_images[:,t]
        tgt_label_valid = tgt_label
        warp_ref_img_t = warp_ref_img[:, t]
        warp_ref_lmark_t = warp_ref_lmark[:, t]
        prevs = [prevs[0], prevs[2]] # prev_label and prev_fake_image
        prevs = [prev.contiguous().view(b, -1, h, w) if prev is not None else None for prev in prevs]
        ani_img_t = ani_img[:, t] if self.opt.warp_ani and ani_img is not None else None
        ani_lmark_t = ani_lmark[:, t] if self.opt.warp_ani and ani_lmark is not None else None

        return tgt_label, tgt_label_valid, tgt_image, warp_ref_img_t, warp_ref_lmark_t, ani_img_t, ani_lmark_t, prevs

    def concat_prev(self, prev, now):
        if type(prev) == list:
            return [self.concat_prev(p, n) for p, n in zip(prev, now)]
        if prev is None:
            prev = now.unsqueeze(1).repeat(1, self.opt.n_frames_G-1, 1, 1, 1)
        else:
            prev = torch.cat([prev[:, 1:], now.unsqueeze(1)], dim=1)
        return prev.detach()
    
    # def crop_template(self, image, template):
    #     b, t, ch, w, h = image.shape
    #     template_array = template.view(-1, w, h).cpu().numpy()
    #     image_temp = image.view(-1, ch, w, h)
    #     cropped_images = []
    #     for temp_id in range(template_array.shape[0]):
    #         cors = np.where(template_array[temp_id] == 1)
    #         left, right = np.min(cors[0]), np.max(cors[0])
    #         up, bottom = np.min(cors[1]), np.max(cors[1])
    #         crop_img = image_temp[temp_id:temp_id+1, :, left:right, up:bottom]
    #         cropped_images.append(F.upsample(crop_img, size=[64, 64]))
        
    #     result = torch.cat(cropped_images).view(b, t, ch, 64, 64)

    #     return result

    def crop_template(self, image, template):
        result = image * template

        return result

    ########################################### inference ###########################################
    def inference(self, tgt_label, ref_labels, ref_images, warp_ref_lmark, warp_ref_img, ani_lmark, ani_img, ref_idx_fix):
        opt = self.opt
        if not self.temporal:
            self.prevs = prevs = [None, None]
            self.t = 0  
        elif not hasattr(self, 'prevs') or self.prevs is None:
            self.prevs = prevs = [None, None]
            self.t = 0
        else:            
            b, _, _, h, w = tgt_label.shape
            prevs = [prev.view(b, -1, h, w) for prev in self.prevs]            
            self.t += 1        
                        
        tgt_label_valid, ref_labels_valid = tgt_label[:,-1], ref_labels

        with torch.no_grad():
            assert self.t == 0
            fake_image, flow, weight, fake_raw_image, warped_image, atn_score, ref_idx, img_ani = self.netG(tgt_label_valid, 
                ref_labels_valid, ref_images, prevs, 
                warp_ref_img, warp_ref_lmark, ani_img, ani_lmark,
                t=self.t, ref_idx_fix=ref_idx_fix)

            ref_label_valid, ref_label, ref_image = self.netG.pick_ref([ref_labels_valid, ref_labels, ref_images], ref_idx)        
            
            if not self.temporal:
                self.prevs = self.concat_prev(self.prevs, [tgt_label_valid, fake_image])            
            
        return fake_image, fake_raw_image, warped_image, flow, weight, atn_score, ref_idx, ref_label, ref_image, img_ani

    def finetune_call_multi(self, tgt_label_list, tgt_image_list, ref_label_list, ref_image_list, warp_ref_lmark_list, warp_ref_img_list, ani_lmark_list=None, ani_img_list=None, iterations=0):
        train_names = ['fc', 'conv_img', 'up']
        params, _ = self.get_train_params(self.netG, train_names)
        self.optimizer_G = self.get_optimizer(params, for_discriminator=False)
        
        update_D = True
        if update_D:
            params = list(self.netD.parameters())       
            self.optimizer_D = self.get_optimizer(params, for_discriminator=True)

        for iteration in tqdm(range(iterations)):
            for data_id in tqdm(range(len(tgt_label_list))):
                tgt_labels, tgt_images, ref_labels, ref_images, warp_ref_lmark, warp_ref_img, ani_lmark, ani_img = \
                    encode_input_finetune(self.opt, data_list=[tgt_label_list[data_id], tgt_image_list[data_id], \
                                                               ref_label_list[data_id], ref_image_list[data_id], \
                                                               warp_ref_lmark_list[data_id], warp_ref_img_list[data_id], \
                                                               ani_lmark_list[data_id] if ani_lmark_list is not None else None, \
                                                               ani_img_list[data_id] if ani_img_list is not None else None],
                                                               dummy_bs=0)

                idx = np.random.randint(tgt_labels.size(1))
                tgt_label, tgt_image = tgt_labels[:,idx].unsqueeze(1), tgt_images[:,idx].unsqueeze(1)
                
                ref_labels_finetune, ref_images_finetune = ref_labels, ref_images
                warp_ref_lmark_finetune, warp_ref_img_finetune = warp_ref_lmark, warp_ref_img
                ani_lmark_finetune, ani_img_finetune = ani_lmark, ani_img

                if ani_img is not None:
                    assert ani_img.shape[1] == ani_lmark.shape[1] == tgt_labels.shape[1]
                    ani_img_finetune = ani_img[:, idx].unsqueeze(1)
                    ani_lmark_finetune = ani_lmark[:, idx].unsqueeze(1)
                if self.opt.n_shot < ref_labels.shape[1]:
                    idxs = np.random.choice(ref_labels.shape[1], self.opt.n_shot)
                    ref_labels_finetune = ref_labels[:, idxs]
                    ref_images_finetune = ref_images[:, idxs]
                if warp_ref_lmark.shape[1] > 1:
                    if self.opt.n_shot >= ref_labels.shape[1]:
                        idxs = np.random.choice(ref_labels.shape[1], self.opt.n_shot)
                    warp_ref_lmark_finetune = warp_ref_lmark[:, idxs[0]].unsqueeze(1)
                    warp_ref_img_finetune = warp_ref_img[:, idxs[0]].unsqueeze(1)

                g_losses, generated, prev, _ = self.forward_generator(tgt_label=tgt_label, tgt_image=tgt_image, \
                                                                        tgt_template=1, tgt_crop_image=None, \
                                                                        flow_gt=[None]*3, conf_gt=[None]*3, \
                                                                        ref_labels=ref_labels_finetune, ref_images=ref_images_finetune, \
                                                                        warp_ref_lmark=warp_ref_lmark_finetune, warp_ref_img=warp_ref_img_finetune, \
                                                                        ani_lmark=ani_lmark_finetune, ani_img=ani_img_finetune)


                g_losses = loss_backward(self.opt, g_losses, self.optimizer_G)

                d_losses = []
                if update_D:
                    d_losses, _ = self.forward_discriminator(tgt_label, tgt_image, 1, ref_labels_finetune, ref_images_finetune, warp_ref_lmark=warp_ref_lmark_finetune, warp_ref_img=warp_ref_img_finetune)
                    d_losses = loss_backward(self.opt, d_losses, self.optimizer_D)

            if (iteration % 10) == 0: 
                message = '(iters: %d) ' % iteration
                loss_dict = dict(zip(self.lossCollector.loss_names, g_losses + d_losses))
                for k, v in loss_dict.items():
                    if v != 0: message += '%s: %.3f ' % (k, v)
                    print(message)

        self.opt.finetune = False