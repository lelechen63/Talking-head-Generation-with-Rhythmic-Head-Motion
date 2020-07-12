from .generator_split import FewShotGenerator, LabelEmbedder
from .warp_module import WarpModule
from models.networks.base_network import BaseNetwork

import torch
import pdb

class SpadeCombineModule(BaseNetwork):
    def __init__(self, opt):
        super().__init__()

        self.netG = FewShotGenerator(opt)
        self.warp = WarpModule(opt, opt.n_frames_G)
        self.get_SPADE_embed(opt)
        self.opt = opt
        self.add_raw_loss = self.opt.add_raw_loss
        self.flow_temp_is_initalized = False

    def forward(self, tgt_lmark, ref_lmarks, ref_imgs, prev, warp_ref_img, warp_ref_lmark, ani_img, ani_lmark, t=0, ref_idx_fix=None):
        # SPADE weight generation
        x, encoded_label, norm_weights, atn, ref_idx \
            = self.netG.weight_generation(ref_imgs, ref_lmarks, tgt_lmark, t=t)

        # get flow and warp
        prev_lmark, prev_img = prev
        flow, weight, img_warp = self.warp.forward_flow(tgt_lmark, warp_ref_lmark, warp_ref_img, ani_lmark, ani_img, prev_lmark, prev_img, ref_idx)

        if self.opt.no_warp:
            img_warp[0] = warp_ref_img if img_warp[0] is not None else None
            img_warp[1] = prev_img if img_warp[1] is not None else None
            img_warp[2] = ani_img if img_warp[2] is not None else None

        # SPADE combine
        ds_ref = [None] * 3
        warp_ref, warp_prev, warp_ani = img_warp
        weight_ref, weight_prev, weight_ani = weight
        has_ref = warp_ref is not None and weight_ref is not None
        has_prev = warp_prev is not None and weight_prev is not None
        has_ani = warp_ani is not None and weight_ani is not None
        if self.warp.warp_ref and has_ref:
            ds_ref[0] = torch.cat([warp_ref, weight_ref], dim=1)
        if self.warp.warp_prev and has_prev: 
            ds_ref[1] = torch.cat([warp_prev, weight_prev], dim=1)
        if self.warp.warp_ani and has_ani: 
            ds_ref[2] = torch.cat([warp_ani, weight_ani], dim=1)

        encoded_label = self.SPADE_combine(encoded_label, ds_ref)

        # add raw
        if self.add_raw_loss:
            encoded_label_raw = [encoded_label[i][0] for i in range(self.netG.n_sc_layers)]
        else: 
            encoded_label_raw = None

        # generate image
        img_final, img_raw = self.netG.img_generation(x, norm_weights, encoded_label, encoded_label_raw)

        return img_final, flow, weight, img_raw, img_warp, atn, ref_idx, None

    # set spade module
    def get_SPADE_embed(self, opt):
        self.img_ref_embedding = LabelEmbedder(opt, opt.output_nc + 1, opt.sc_arch)
        self.img_ani_embedding = LabelEmbedder(opt, opt.output_nc + 1, opt.sc_arch)
        self.img_prev_embedding = None

    ### if using SPADE for combination
    def SPADE_combine(self, encoded_label, ds_ref):                  
        encoded_image_warp = [self.img_ref_embedding(ds_ref[0]),
                              self.img_ani_embedding(ds_ref[2]) if ds_ref[2] is not None else None,
                              self.img_prev_embedding(ds_ref[1]) if ds_ref[1] is not None else None
                             ]
        for i in range(self.netG.n_sc_layers):
            encoded_label[i] = [encoded_label[i]] + [w[i] if w is not None else None for w in encoded_image_warp]

        return encoded_label

    # set temporal
    def set_flow_prev(self):
        self.img_prev_embedding = LabelEmbedder(self.opt, self.opt.output_nc + 1, self.opt.sc_arch)
        self.load_pretrained_net(self.img_ref_embedding, self.img_prev_embedding)
        
        self.warp.set_temporal()
        self.flow_temp_is_initalized = True

    def pick_ref(self, refs, ref_idx):
        return self.warp.pick_ref(refs, ref_idx)

    def trans_init_G(self):
        self.netG.encoder_init()