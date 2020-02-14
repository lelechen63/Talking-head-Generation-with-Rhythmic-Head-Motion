from .generator_split import FewShotGenerator
from .warp_module import WarpModule
from models.networks.base_network import BaseNetwork

import pdb

class LinearCombineModule(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.netG = FewShotGenerator(opt)
        self.warp = WarpModule(opt, opt.n_frames_G)

    def forward(self, tgt_lmark, ref_lmarks, ref_imgs, prev, warp_ref_img, warp_ref_lmark, ani_img, ani_lmark, t=0, ref_idx_fix=None):
        # generate image
        fake_raw_img, atn, ref_idx = self.netG(label=tgt_lmark, label_refs=ref_lmarks, img_refs=ref_imgs, t=t)

        # warp and linear combine
        prev_lmark, prev_img = prev
        img_final, flow, weight, img_warp, img_ani = self.warp(fake_raw_img, tgt_lmark, warp_ref_lmark, warp_ref_img, ani_lmark, ani_img, prev_lmark, prev_img)

        return img_final, flow, weight, fake_raw_img, img_warp, atn, ref_idx, img_ani

    def pick_ref(self, refs, ref_idx):
        return self.warp.pick_ref(refs, ref_idx)