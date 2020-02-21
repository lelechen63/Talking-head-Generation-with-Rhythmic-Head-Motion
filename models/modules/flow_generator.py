import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
import copy
import pdb

from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import SPADEResnetBlock

class FlowGenerator(BaseNetwork):
    def __init__(self, opt, n_frames_G):
        super().__init__()
        self.opt = opt                
        input_nc = (opt.label_nc if opt.label_nc != 0 else opt.input_nc) * n_frames_G
        input_nc += opt.output_nc * (n_frames_G - 1)        
        nf = opt.nff
        n_blocks = opt.n_blocks_F
        n_downsample_F = opt.n_downsample_F
        self.flow_multiplier = opt.flow_multiplier        
        nf_max = 1024
        ch = [min(nf_max, nf * (2 ** i)) for i in range(n_downsample_F + 1)]
                
        norm = opt.norm_F
        norm_layer = get_nonspade_norm_layer(opt, norm)
        activation = nn.LeakyReLU(0.2, True)
        
        down_flow = [norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, padding=1)), activation]        
        for i in range(n_downsample_F):            
            down_flow += [norm_layer(nn.Conv2d(ch[i], ch[i+1], kernel_size=3, padding=1, stride=2)), activation]            
                   
        ### resnet blocks
        res_flow = []
        ch_r = min(nf_max, nf * (2**n_downsample_F))        
        for i in range(n_blocks):
            res_flow += [SPADEResnetBlock(ch_r, ch_r, norm=norm)]
    
        ### upsample
        up_flow = []                         
        for i in reversed(range(n_downsample_F)):
            if opt.flow_deconv:
                up_flow += [norm_layer(nn.ConvTranspose2d(ch[i+1], ch[i], kernel_size=3, stride=2, padding=1, output_padding=1)), activation]
            else:
                up_flow += [nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(ch[i+1], ch[i], kernel_size=3, padding=1)), activation]
                              
        conv_flow = [nn.Conv2d(nf, 2, kernel_size=3, padding=1)]
        conv_w = [nn.Conv2d(nf, 1, kernel_size=3, padding=1), nn.Sigmoid()] 
      
        self.down_flow = nn.Sequential(*down_flow)        
        self.res_flow = nn.Sequential(*res_flow)
        self.up_flow = nn.Sequential(*up_flow)
        self.conv_flow = nn.Sequential(*conv_flow)        
        self.conv_w = nn.Sequential(*conv_w)

    def forward(self, label, label_prev, img_prev, for_ref=False):
        label = torch.cat([label, label_prev, img_prev], dim=1)
        downsample = self.down_flow(label)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)        
        flow = self.conv_flow(flow_feat) * self.flow_multiplier
        weight = self.conv_w(flow_feat)
        return flow, weight