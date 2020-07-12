from models.modules.flow_generator import FlowGenerator
from models.networks.base_network import BaseNetwork
import torch

import pdb

class WarpModule(BaseNetwork):
    def __init__(self, opt, n_frames_G):
        super().__init__()
        self.opt = opt
        self.warp_ref = opt.warp_ref
        self.warp_ani = opt.warp_ani
        self.warp_prev = False

        # warp reference
        if self.warp_ref:
            self.flow_network_ref = FlowGenerator(opt, 2)
        
        # warp animation
        if self.warp_ani:
            self.flow_network_ani = FlowGenerator(opt, 2)

    # set previous warper
    def set_temporal(self):
        self.warp_prev = True
        if self.opt.same_flownet and self.warp_ref:
            print('use same flownet for temporal and reference')
            self.flow_network_temp = self.flow_network_ref
        else:
            self.flow_network_temp = FlowGenerator(self.opt, self.opt.n_frames_G)
            if self.warp_ref:
                self.load_pretrained_net(self.flow_network_ref, self.flow_network_temp)

    # [target landmark, reference landmark, reference image] -> flow, weight, warp -> combine
    def forward(self, fake_raw_img, tgt_lmark, ref_lmarks, ref_imgs, ani_lmark, ani_img, prev_lmark=None, prev_img=None, ref_idx=None):
        # get flow and warp
        flow, weight, img_warp = self.forward_flow(tgt_lmark, ref_lmarks, ref_imgs, ani_lmark, ani_img, prev_lmark, prev_img, ref_idx)

        # linear combine
        img_final, img_ani = self.forward_linear(flow, weight, img_warp, fake_raw_img)

        return img_final, flow, weight, img_warp, img_ani

    # get flow
    def forward_flow(self, tgt_lmark, ref_lmarks, ref_imgs, ani_lmark, ani_img, prev_lmark, prev_img, ref_idx):
        has_ref = ref_lmarks is not None and ref_imgs is not None
        has_prev = prev_lmark is not None and prev_img is not None
        has_ani = ani_lmark is not None and ani_img is not None

        flow, weight, img_warp = [None] * 3, [None] * 3, [None] * 3
        if self.warp_ref and has_ref:
            # select most similar reference image index by ref_idx
            flow_ref, weight_ref = self.flow_network_ref(tgt_lmark, ref_lmarks, ref_imgs, for_ref=True)
            img_ref_warp = self.resample(ref_imgs, flow_ref)
            flow[0], weight[0], img_warp[0] = flow_ref, weight_ref, img_ref_warp[:,:3]

        if self.warp_prev and has_prev:
            flow_prev, weight_prev = self.flow_network_temp(tgt_lmark, prev_lmark, prev_img)
            img_prev_warp = self.resample(prev_img[:,-3:], flow_prev)
            flow[1], weight[1], img_warp[1] = flow_prev, weight_prev, img_prev_warp

        if self.warp_ani and has_ani:
            flow_ani, weight_ani = self.flow_network_ani(tgt_lmark, ani_lmark, ani_img)
            img_ani_warp = self.resample(ani_img[:,-3:], flow_ani)
            flow[2], weight[2], img_warp[2] = flow_ani, weight_ani, img_ani_warp

        return flow, weight, img_warp    


    # warp
    def forward_linear(self, flow, weight, img_warp, img_raw):
        weight_ref, weight_prev, weight_ani= weight
        img_ref_warp, img_prev_warp, img_ani_warp = img_warp            
        
        has_ref = weight_ref is not None and img_ref_warp is not None
        has_prev = weight_prev is not None and img_prev_warp is not None
        has_ani = weight_ani is not None and img_ani_warp is not None

        # combine raw result with animation image
        img_ani = None
        img_final = img_raw
        if self.warp_ani and has_ani:
            img_final = img_final * weight_ani + img_ani_warp * (1 - weight_ani)
            img_ani = img_final

        # combine raw result with reference image
        if self.warp_ref and has_ref:
            img_final = img_final * weight_ref + img_ref_warp * (1 - weight_ref)        

        ### combine generated frame with previous frame
        if self.warp_prev and has_prev:
            img_final = img_final * weight_prev + img_prev_warp * (1 - weight_prev)        

        return img_final, img_ani

    
    # pick the reference image that is most similar to current frame
    def pick_ref(self, refs, ref_idx):
        if type(refs) == list:
            return [self.pick_ref(r, ref_idx) for r in refs]
        if ref_idx is None:
            return refs[:,0]
        ref_idx = ref_idx.long().view(-1, 1, 1, 1, 1)
        ref = refs.gather(1, ref_idx.expand_as(refs)[:,0:1])[:,0]        
        return ref
