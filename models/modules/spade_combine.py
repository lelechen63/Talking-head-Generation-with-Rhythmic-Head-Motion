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

        encoded_label = self.SPADE_combine(encoded_label, ds_ref)

    