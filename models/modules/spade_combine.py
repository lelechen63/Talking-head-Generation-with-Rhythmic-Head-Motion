from .generator import FewShotGenerator, LabelEmbedder
from .warp_module import WarpModule
from models.networks.base_network import BaseNetwork

class SpadeCombineModule(BaseNetwork):
    def __init__(self, opt, n_frames_G):
        self.netG = FewShotGenerator(opt)
        self.warp = WarpModule(opt, opt.n_frames_G)

    def forward(self, fake_raw_img, tgt_lmark, ref_lmarks, ref_imgs, prev_lmark=None, prev_img=None, ref_idx=None, t=0):
        # get flow and warp
        flow, weight, img_warp = self.warp.forward_flow(tgt_lmark, ref_lmarks, ref_imgs, prev_lmark, prev_img, ref_idx)

        # SPADE weight generation
        x, encoded_label, norm_weights, atn, ref_idx \
            = self.gen.weight_generation(ref_imgs, ref_lmarks, tgt_lmark, t=t)

        # SPADE combine
        ds_ref = [None] * 2
        warp_ref, warp_prev = img_warp
        weight_ref, weight_prev = weight
        has_ref = warp_ref is not None and weight_ref is not None
        has_prev = warp_prev is not None and weight_prev is not None
        if self.warp.warp_ref and has_ref:
            ds_ref[0] = torch.cat([warp_ref, weight_ref], dim=1)
        if self.warp_prev and has_prev: 
            ds_ref[1] = torch.cat([warp_prev, weight_prev], dim=1)

        encoded_label = self.SPADE_combine(encoded_label, ds_ref)

        # generate image
        img_final = self.gen.img_generation(x, norm_weights, encoded_label)

        return img_final, flow, weight, None, img_warp, atn, ref_idx

    ### if using SPADE for combination
    def SPADE_combine(self, encoded_label, ds_ref):        
        if self.spade_combine:            
            encoded_image_warp = [self.img_ref_embedding(ds_ref[0]), 
                                  self.img_prev_embedding(ds_ref[1]) if ds_ref[1] is not None else None]
            for i in range(self.n_sc_layers):
                encoded_label[i] = [encoded_label[i]] + [w[i] if w is not None else None for w in encoded_image_warp]
        return encoded_label