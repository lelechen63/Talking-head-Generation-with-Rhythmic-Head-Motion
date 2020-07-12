from .generator_split import FewShotGenerator
from .warp_module import WarpModule
from models.networks.base_network import BaseNetwork

import pdb

class LinearCombineModule(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.netG = FewShotGenerator(opt)
        self.warp = WarpModule(opt, opt.n_frames_G)
        self.flow_temp_is_initalized = False
        self.opt = opt

    def forward(self, tgt_lmark, ref_lmarks, ref_imgs, prev, warp_ref_img, warp_ref_lmark, ani_img, ani_lmark, t=0, ref_idx_fix=None):
        # generate image
        fake_raw_img, atn, ref_idx = self.netG(label=tgt_lmark, label_refs=ref_lmarks, img_refs=ref_imgs, t=t)
        if self.opt.origin_ref_select:
            warp_ref_lmark, warp_ref_img = self.warp.pick_ref([ref_lmarks, ref_imgs], ref_idx)

        # warp and linear combine
        no_warp = not (self.warp.warp_ani or self.warp.warp_prev or self.warp.warp_ref)
        if not no_warp:
            prev_lmark, prev_img = prev
            img_final, flow, weight, img_warp, img_ani = self.warp(fake_raw_img, tgt_lmark, warp_ref_lmark, warp_ref_img, ani_lmark, ani_img, prev_lmark, prev_img, ref_idx=ref_idx)

            # no warping for image
            if self.opt.no_warp:
                img_warp[0] = warp_ref_img if img_warp[0] is not None else None
                img_warp[1] = prev_img if img_warp[1] is not None else None
                img_warp[2] = ani_img if img_warp[2] is not None else None

        else:
            img_final = fake_raw_img
            flow = [None] * 3
            weight = [None] * 3
            img_warp = [None] * 3
            img_ani = None
            fake_raw_img = None

        return img_final, flow, weight, fake_raw_img, img_warp, atn, ref_idx, img_ani

    def pick_ref(self, refs, ref_idx):
        return self.warp.pick_ref(refs, ref_idx)

    def set_flow_prev(self):
        self.warp.set_temporal()
        self.flow_temp_is_initalized = True

    def trans_init_G(self):
        self.netG.encoder_init()