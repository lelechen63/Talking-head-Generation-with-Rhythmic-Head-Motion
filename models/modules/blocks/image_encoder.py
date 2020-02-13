import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
import copy
import pdb

from models.networks.base_network import BaseNetwork, batch_conv
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import SPADEResnetBlock, SPADEConv2d, actvn
import torch.nn.utils.spectral_norm as sn

class ImageEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()


    def attention_module(self, x, label, label_ref, attention=None):
        b, c, h, w = x.size()
        n = self.opt.n_shot
        b = b//n
        x = x.view(b, n, c, h*w)
        
        if attention is None:
            atn_key = self.atn_key_first(label_ref)
            atn_query = self.atn_query_first(label)                
            
            for i in range(self.n_downsample_A):            
                atn_key = getattr(self, 'atn_key_'+str(i))(atn_key)
                atn_query = getattr(self, 'atn_query_'+str(i))(atn_query)
            
            atn_key = atn_key.view(b, n, c, -1)
            atn_query = atn_query.view(b, 1, c, -1).expand_as(atn_key)            
                        
            energy = torch.sum(atn_key * atn_query, dim=2)
            attention = nn.Softmax(dim=1)(energy) # b X n X hw
        else:
            attention = attention.view(b, n, h*w)

        out = torch.sum(x * attention.unsqueeze(2).expand_as(x), dim=1).view(b, c, h, w)
        
        return out, attention.view(b, n, h, w)        

    ### encode the reference image to get features for weight generation
    def reference_encoding(self, img_ref, label_ref, label, n, t=0):
        # apply conv to both reference label and image, then multiply them together for encoding
        x = self.ref_img_first(img_ref)
        x_label = self.ref_label_first(label_ref)

        atn = ref_idx = None # attention map and the index of the most similar reference image
        for i in range(self.n_downsample_G):            
            x = getattr(self, 'ref_img_down_'+str(i))(x)
            x_label = getattr(self, 'ref_label_down_'+str(i))(x_label)

            ### combine different reference images at a particular layer if n_shot > 1
            if n > 1 and i == self.n_downsample_A - 1:
                x, atn = self.attention_module(x, label, label_ref)
                x_label, _ = self.attention_module(x_label, None, None, atn)

                atn_sum = atn.view(label.shape[0], n, -1).sum(2)
                ref_idx = torch.argmax(atn_sum, dim=1)                
        
        # get all corresponding layers in the encoder output for generating weights in corresponding layers
        encoded_ref = None
        if self.opt.isTrain or n > 1 or t == 0:
            encoded_image_ref = [x]   
            encoded_label_ref = [x_label]       
            
            for i in reversed(range(self.n_downsample_G)):
                conv = getattr(self, 'ref_img_up_'+str(i))(encoded_image_ref[-1])
                encoded_image_ref.append(conv)

                conv_label = getattr(self, 'ref_label_up_'+str(i))(encoded_label_ref[-1])            
                encoded_label_ref.append(conv_label)
            
            encoded_ref = []
            for i in range(len(encoded_image_ref)):  
                conv, conv_label = encoded_image_ref[i], encoded_label_ref[i]
                b, c, h, w = conv.size()
                conv_label = nn.Softmax(dim=1)(conv_label)        
                conv_prod = (conv.view(b, c, 1, h*w) * conv_label.view(b, 1, c, h*w)).sum(3, keepdim=True)                    
                encoded_ref.append(conv_prod)

            encoded_ref = encoded_ref[::-1]

        return x, encoded_ref, atn, ref_idx