# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
import copy
import pdb

# demo used
import util.util as util

from models.networks.base_network import BaseNetwork, batch_conv
from models.networks.architecture import SPADEResnetBlockConcat, SPADEResnetBlock, SPADEConv2d, actvn
import torch.nn.utils.spectral_norm as sn

# generator
class FewShotGenerator(BaseNetwork):    
    def __init__(self, opt):
        super().__init__()
        ########################### define params ##########################
        self.opt = opt
        self.add_raw_loss = opt.add_raw_loss and opt.spade_combine
        self.n_downsample_G = n_downsample_G = opt.n_downsample_G # number of downsamplings in generator
        self.n_downsample_A = opt.n_downsample_A # number of downsamplings in attention module        
        self.nf = nf = opt.ngf                                    # base channel size
        
        nf_max = min(1024, nf * (2**n_downsample_G))
        self.ch = ch = [min(nf_max, nf * (2 ** i)) for i in range(n_downsample_G + 2)]
                
        ### SPADE          
        self.norm = norm = opt.norm_G
        self.conv_ks = conv_ks = opt.conv_ks    # conv kernel size in main branch
        self.embed_ks = opt.embed_ks # conv kernel size in embedding network
        self.spade_ks = spade_ks = opt.spade_ks # conv kernel size in SPADE
        self.spade_combine = opt.spade_combine  # combine ref/prev frames with current using SPADE
        self.n_sc_layers = opt.n_sc_layers      # number of layers to perform spade combine        
        ch_hidden = []                          # hidden channel size in SPADE module
        for i in range(n_downsample_G + 1):
            ch_hidden += [[ch[i]]] if not self.spade_combine or i >= self.n_sc_layers else [[ch[i]]*4]
        self.ch_hidden = ch_hidden

        ### adaptive SPADE / Convolution
        self.adap_spade = opt.adaptive_spade                               # use adaptive weight generation for SPADE
        self.adap_embed = opt.adaptive_spade and not opt.no_adaptive_embed # use adaptive for the label embedding network      
        self.n_adaptive_layers = opt.n_adaptive_layers if opt.n_adaptive_layers != -1 else n_downsample_G  # number of adaptive layers

        # weight generation
        self.n_fc_layers = opt.n_fc_layers    # number of fc layers in weight generation            

        ########################### define network ##########################
        norm_ref = norm.replace('spade', '')
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc  #1
        ref_nc = opt.output_nc #3
        if not opt.use_new or opt.transfer_initial:
            self.image_encoder = Encoder(norm_ref = norm_ref, 
                                        nf=self.nf, ch=self.ch, 
                                        n_shot=self.opt.n_shot, 
                                        n_downsample_G=self.n_downsample_G, 
                                        n_downsample_A=self.n_downsample_A, 
                                        isTrain=self.opt.isTrain, 
                                        ref_nc=ref_nc)
            
            self.lmark_encoder = Encoder(norm_ref = norm_ref, 
                                        nf=self.nf, ch=self.ch, 
                                        n_shot=self.opt.n_shot, 
                                        n_downsample_G=self.n_downsample_G, 
                                        n_downsample_A=self.n_downsample_A, 
                                        isTrain=self.opt.isTrain, 
                                        ref_nc=input_nc)
        if opt.use_new:
            self.encoder = EncoderSelfAtten(norm_ref = norm_ref, 
                                        nf=self.nf, ch=self.ch, 
                                        n_shot=self.opt.n_shot, 
                                        n_downsample_G=self.n_downsample_G, 
                                        n_downsample_A=self.n_downsample_A, 
                                        isTrain=self.opt.isTrain, 
                                        ref_nc=ref_nc,
                                        lmark_nc=input_nc)

        self.comb_encoder = CombEncoder(norm_ref=norm_ref, ch=self.ch,
                                        n_shot=self.opt.n_shot,
                                        n_downsample_G=self.n_downsample_G)

        if not opt.no_atten:
            self.atten_gen = AttenGen(norm_ref=norm_ref, input_nc=input_nc, 
                                  nf=self.nf, ch=self.ch,
                                  n_shot=self.opt.n_shot,
                                  n_downsample_A=self.n_downsample_A)

        ### SPADE / main branch weight generation
        if self.adap_spade:
            self.weight_gen = WeightGen(ch_hidden=self.ch_hidden, embed_ks=self.embed_ks,
                                        spade_ks=self.spade_ks, n_fc_layers=self.n_fc_layers,
                                        n_adaptive_layers=self.n_adaptive_layers,
                                        ch=self.ch,
                                        adap_embed=self.adap_embed)

        ### label embedding network 
        self.label_embedding = LabelEmbedder(opt, input_nc, opt.netS, 
            params_free_layers=(self.n_adaptive_layers if self.adap_embed else 0))
            
        ### main branch layers
        for i in reversed(range(n_downsample_G + 1)):
            hidden_nc = ch_hidden[i]
            if i >= self.n_sc_layers or not opt.use_new:
                setattr(self, 'up_%d' % i, SPADEResnetBlock(ch[i+1], ch[i], norm=norm, hidden_nc=hidden_nc, 
                        conv_ks=conv_ks, spade_ks=spade_ks,
                        conv_params_free=False,
                        norm_params_free=(self.adap_spade and i < self.n_adaptive_layers)))
            else:
                setattr(self, 'up_%d' % i, SPADEResnetBlockConcat(ch[i+1], ch[i], norm=norm, hidden_nc=hidden_nc, 
                        conv_ks=conv_ks, spade_ks=spade_ks,
                        conv_params_free=False,
                        norm_params_free=(self.adap_spade and i < self.n_adaptive_layers)))

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        self.up = functools.partial(F.interpolate, scale_factor=2)


    def forward(self, label, label_refs, img_refs, t=0):
        # SPADE weight generation
        x, encoded_label, norm_weights, atn, ref_idx \
            = self.weight_generation(img_refs, label_refs, label, t=t)        

        # main branch convolution layers
        fake_raw_img, _ = self.img_generation(x, norm_weights, encoded_label)

        return fake_raw_img, atn, ref_idx

    # generate image
    def img_generation(self, x, norm_weights, encoded_label, encoded_label_raw=None):
        # main branch convolution layers
        for i in range(self.n_downsample_G, -1, -1):          
            conv_weight = None
            norm_weight = norm_weights[i] if (self.adap_spade and i < self.n_adaptive_layers) else None
            # if require loss for raw image
            if self.add_raw_loss and i < self.n_sc_layers:
                if i == self.n_sc_layers - 1: x_raw = x
                x_raw = getattr(self, 'up_'+str(i))(x_raw, encoded_label_raw[i], conv_weights=conv_weight, norm_weights=norm_weight)    
                if i != 0: x_raw = self.up(x_raw)
            x = getattr(self, 'up_'+str(i))(x, encoded_label[i], conv_weights=conv_weight, norm_weights=norm_weight)
            if i != 0: x = self.up(x)

        # raw synthesized image
        x = self.conv_img(actvn(x))
        fake_raw_img = torch.tanh(x)
        x_raw = None if not self.add_raw_loss else torch.tanh(self.conv_img(actvn(x_raw)))

        return fake_raw_img, x_raw

    ### generate weights based on the encoded features
    def weight_generation(self, img_ref, label_ref, label, t=0):
        b, n, c, h, w = img_ref.size()
        img_ref, label_ref = img_ref.view(b*n, -1, h, w), label_ref.view(b*n, -1, h, w)

        # encode image and landmark
        x, encoded_ref, atn, ref_idx = self.reference_encoding(img_ref, label_ref, label, n, t)
        
        # get weight for model
        if self.opt.isTrain or n > 1 or t == 0:
            if self.adap_spade:
                embedding_weights, norm_weights = self.weight_gen(encoded_ref)
            else:
                embedding_weights, norm_weights = [], []

            if not self.opt.isTrain:
                self.embedding_weights, self.norm_weights = embedding_weights, norm_weights
        else:
            embedding_weights, norm_weights = self.embedding_weights, self.norm_weights
        
        # encode target landmark
        encoded_label = self.label_embedding(label, weights=(embedding_weights if self.adap_embed else None))        

        return x, encoded_label, norm_weights, atn, ref_idx

    ### encode the reference image to get features for weight generation
    def reference_encoding(self, img_ref, label_ref, label, n, t=0):
        # get attention
        if self.opt.no_atten:
            atten = None
            ref_idx = None
        else:
            atten, ref_idx = self.atten_gen(label, label_ref)    # b x n x hw

        # encode image and landmarks separately
        if not self.opt.use_new:
            x = self.image_encoder(img_ref, atten)
            x_label = self.lmark_encoder(label_ref, atten)
        else:
            x, x_label = self.encoder(img_ref, label_ref, atten)

        # combine image and landmarks for spade weight
        if self.opt.isTrain or self.opt.n_shot > 1 or t == 0:
            encoded_ref = self.comb_encoder(x, x_label, t)
        else:
            encoded_ref = None

        return x, encoded_ref, atten, ref_idx

    ### pick the reference image that is most similar to current frame
    def pick_ref(self, refs, ref_idx):
        if type(refs) == list:
            return [self.pick_ref(r, ref_idx) for r in refs]
        if ref_idx is None:
            return refs[:,0]
        ref_idx = ref_idx.long().view(-1, 1, 1, 1, 1)
        ref = refs.gather(1, ref_idx.expand_as(refs)[:,0:1])[:,0]        
        return ref

    # transfer parameters from image encoder and landmark encoder to attention encoder
    def encoder_init(self):
        # transfer
        self.encoder.conv1 = self.image_encoder.conv1
        self.encoder.conv2 = self.lmark_encoder.conv1
        for i in range(self.n_downsample_G):
            setattr(self.encoder, 'ref_down_img_%d' % i, getattr(self.image_encoder, 'ref_down_%d' % i))
            setattr(self.encoder, 'ref_down_lmark_%d' % i, getattr(self.lmark_encoder, 'ref_down_%d' % i))

        # delete
        self.image_encoder = None
        self.lmark_encoder = None

# encode image or landmarks
class Encoder(BaseNetwork):
    def __init__(self, norm_ref, nf, ch, n_shot, n_downsample_G, n_downsample_A, isTrain, ref_nc=3):
        super().__init__()

        # parameters for model
        self.conv1 = SPADEConv2d(ref_nc, nf, norm=norm_ref)

        for i in range(n_downsample_G):
            ch_in, ch_out = ch[i], ch[i+1]            
            setattr(self, 'ref_down_%d' % i, SPADEConv2d(ch_in, ch_out, stride=2, norm=norm_ref))

        # other parameters
        self.isTrain = isTrain
        self.n_shot = n_shot
        self.n_downsample_G = n_downsample_G
        self.n_downsample_A = n_downsample_A
        
    # x:[b*n, c, h, w], atten:[b, n, h*w]
    def forward(self, x, atten):
        # prepare
        n = self.n_shot
        assert x.shape[0] % n == 0

        # forward
        x = self.conv1(x)
        for i in range(self.n_downsample_G):
            x = getattr(self, 'ref_down_'+str(i))(x)

            # attention
            if n > 1 and i == self.n_downsample_A-1:
                b, c, h, w = x.shape
                x = x.view(b//n, n, c, h*w)

                if atten is None:
                    x = torch.mean(x, dim=1).view(b//n, c, h, w)
                else:
                    x = torch.sum(x * atten.unsqueeze(2).expand_as(x), dim=1).view(b//n, c, h, w)

        return x

# encode image and landmark together
class EncoderSelfAtten(BaseNetwork):
    def __init__(self, norm_ref, nf, ch, n_shot, n_downsample_G, n_downsample_A, isTrain, ref_nc=3, lmark_nc=1):
        super().__init__()

        # parameters for model
        self.conv1 = SPADEConv2d(ref_nc, nf, norm=norm_ref)
        self.conv2 = SPADEConv2d(lmark_nc, nf, norm=norm_ref)

        for i in range(n_downsample_G):
            ch_in, ch_out = ch[i], ch[i+1]            
            setattr(self, 'ref_down_img_%d' % i, SPADEConv2d(ch_in, ch_out, stride=2, norm=norm_ref))
            setattr(self, 'ref_down_lmark_%d' % i, SPADEConv2d(ch_in, ch_out, stride=2, norm=norm_ref))
            if n_shot > 1 and i == n_downsample_A-1:
                self.fusion1 = SPADEConv2d(ch_out*2, ch_out, norm=norm_ref)
                self.fusion2 = SPADEConv2d(ch_out*2, ch_out, norm=norm_ref)
                self.fusion = SPADEResnetBlock(ch_out*2, ch_out, norm=norm_ref)
                self.atten1 = SPADEConv2d(ch_out*n_shot, ch_out, norm=norm_ref)
                self.atten2 = SPADEConv2d(ch_out*n_shot, ch_out, norm=norm_ref)
                
        # other parameters
        self.isTrain = isTrain
        self.n_shot = n_shot
        self.n_downsample_G = n_downsample_G
        self.n_downsample_A = n_downsample_A
        
    # x:[b*n, c, h, w], atten:[b, n, h*w]
    def forward(self, x, x_label, atten):
        # prepare
        n = self.n_shot
        assert x.shape[0] % n == 0
        assert x_label.shape[0] % n == 0

        # forward
        x = self.conv1(x)
        x_label = self.conv2(x_label)
        for i in range(self.n_downsample_G):
            x = getattr(self, 'ref_down_img_'+str(i))(x)
            x_label = getattr(self, 'ref_down_lmark_'+str(i))(x_label)
            # attention
            if n > 1 and i == self.n_downsample_A-1:
                x, x_label = self.attention_module(x, x_label, atten, n)

        return x, x_label

    # attention
    def attention_module(self, x, x_label, atten, n):
        b, c, h, w = x.shape
        ##### cross attention #####(TODO: check dimension)
        x_atten_cat = torch.cat([x, x_label], axis=1)
        x_atten_cat = self.fusion(x_atten_cat)
        #### self attention #####
        x_self_atten = x.view(b//n, n*c, h, w)
        x_self_atten = self.atten1(x_self_atten)
        x_label_self_atten = x_label.view(b//n, n*c, h, w)
        x_label_self_atten = self.atten2(x_label_self_atten)
        # fuse to two branch
        x_cat = torch.cat([x, x_atten_cat], axis=1)
        x = self.fusion1(x_cat)
        x_label_cat = torch.cat([x_label, x_atten_cat], axis=1)
        x_label = self.fusion2(x_label_cat)
        ##### ref & tgt attention #####
        x = x.view(b//n, n, c, h*w)
        x = torch.sum(x * atten.unsqueeze(2).expand_as(x), dim=1).view(b//n, c, h, w) + x_self_atten
        x_label = x_label.view(b//n, n, c, h*w)
        x_label = torch.sum(x_label * atten.unsqueeze(2).expand_as(x_label), dim=1).view(b//n, c, h, w) + x_label_self_atten

        return x, x_label

# encode combining image and landmarks
class CombEncoder(BaseNetwork):
    def __init__(self, norm_ref, ch, n_shot, n_downsample_G):
        super().__init__()

        # parameters for model
        for i in range(n_downsample_G):
            ch_in, ch_out = ch[i], ch[i+1]                              
            setattr(self, 'ref_img_up_%d' % i, SPADEConv2d(ch_out, ch_in, norm=norm_ref))
            setattr(self, 'ref_label_up_%d' % i, SPADEConv2d(ch_out, ch_in, norm=norm_ref))  

        # other parameter
        self.n_downsample_G = n_downsample_G
        self.n_shot = n_shot

    def forward(self, x, x_label, t):
        encoded_image_ref = x   
        encoded_label_ref = x_label       
    
        encoded_ref = []
        for i in reversed(range(self.n_downsample_G)):
            b, c, h, w = encoded_image_ref.size()

            # attention multiply
            conv_label = nn.Softmax(dim=1)(encoded_label_ref)
            conv_prod = (encoded_image_ref.view(b, c, 1, h*w) * conv_label.view(b, 1, c, h*w)).sum(3, keepdim=True)
            encoded_ref.append(conv_prod)

            # forward
            encoded_image_ref = getattr(self, 'ref_img_up_'+str(i))(encoded_image_ref)
            encoded_label_ref = getattr(self, 'ref_label_up_'+str(i))(encoded_label_ref)            

        # last one
        b, c, h, w = encoded_image_ref.size()

        # attention multiply
        conv_label = nn.Softmax(dim=1)(encoded_label_ref)
        conv_prod = (encoded_image_ref.view(b, c, 1, h*w) * conv_label.view(b, 1, c, h*w)).sum(3, keepdim=True)
        encoded_ref.append(conv_prod)

        # reverse
        encoded_ref = encoded_ref[::-1]

        return encoded_ref

# generate attention
class AttenGen(BaseNetwork):
    def __init__(self, norm_ref, input_nc, nf, ch, n_shot, n_downsample_A):
        super().__init__()

        # parameters for model
        self.atn_query_first = SPADEConv2d(input_nc, nf, norm=norm_ref)
        self.atn_key_first = SPADEConv2d(input_nc, nf, norm=norm_ref)
        for i in range(n_downsample_A):
            f_in, f_out = ch[i], ch[i+1]
            setattr(self, 'atn_key_%d' % i, SPADEConv2d(f_in, f_out, stride=2, norm=norm_ref))
            setattr(self, 'atn_query_%d' % i, SPADEConv2d(f_in, f_out, stride=2, norm=norm_ref))

        # other parameters
        self.n_shot = n_shot
        self.n_downsample_A = n_downsample_A

    def forward(self, label, label_ref):
        n = self.n_shot
        assert label_ref.shape[0] % n == 0

        atn_key = self.atn_key_first(label_ref)
        atn_query = self.atn_query_first(label)                
        
        for i in range(self.n_downsample_A):            
            atn_key = getattr(self, 'atn_key_'+str(i))(atn_key)
            atn_query = getattr(self, 'atn_query_'+str(i))(atn_query)
        
        b, c, h, w = atn_key.size()
        b_n = b//n

        atn_key = atn_key.view(b_n, n, c, -1)
        atn_query = atn_query.view(b_n, 1, c, -1).expand_as(atn_key)            
                    
        energy = torch.sum(atn_key * atn_query, dim=2)
        attention = nn.Softmax(dim=1)(energy) # b X n X hw

        # get most similar reference index
        atn_sum = attention.view(b_n, n, -1).sum(2)
        ref_idx = torch.argmax(atn_sum, dim=1)

        return attention, ref_idx

# generate weight for model
class WeightGen(BaseNetwork):
    def __init__(self, ch_hidden, embed_ks, spade_ks, n_fc_layers, n_adaptive_layers, ch, adap_embed):
        super().__init__()

        # parameters for model
        for i in range(n_adaptive_layers):
            ch_in, ch_out = ch[i], ch[i+1]
            embed_ks2 = embed_ks**2
            spade_ks2 = spade_ks**2
            ch_h = ch_hidden[i][0]

            fc_names, fc_outs = [], []
            fc0_out = fcs_out = (ch_h * spade_ks2 + 1) * 2
            fc1_out = (ch_h * spade_ks2 + 1) * (1 if ch_in != ch_out else 2)
            fc_names += ['fc_spade_0', 'fc_spade_1', 'fc_spade_s']
            fc_outs += [fc0_out, fc1_out, fcs_out]
            if adap_embed:                        
                fc_names += ['fc_spade_e']
                fc_outs += [ch_in * embed_ks2 + 1]

            # define weight for fully connected layers
            for n, l in enumerate(fc_names):
                fc_in = ch_out
                fc_layer = [sn(nn.Linear(fc_in, ch_out))]
                for k in range(1, n_fc_layers): 
                    fc_layer += [sn(nn.Linear(ch_out, ch_out))]
                fc_layer += [sn(nn.Linear(ch_out, fc_outs[n]))]
                setattr(self, '%s_%d' % (l, i), nn.Sequential(*fc_layer))

        # other parameters
        self.ch = ch
        self.ch_hidden = ch_hidden
        self.embed_ks = embed_ks
        self.spade_ks = spade_ks
        self.adap_embed = adap_embed
        self.n_adaptive_layers = n_adaptive_layers

    def forward(self, encoded_ref):
        embedding_weights, norm_weights = [], []          
        for i in range(self.n_adaptive_layers):
            feat = encoded_ref[min(len(encoded_ref)-1, i+1)]                         
            embedding_weight, norm_weight = self.get_SPADE_weights(feat, i) 
            embedding_weights.append(embedding_weight)
            norm_weights.append(norm_weight)

        return embedding_weights, norm_weights

    ### adaptively generate weights for SPADE in layer i of generator
    def get_SPADE_weights(self, x, i):
        ch_in, ch_out = self.ch[i], self.ch[i+1]
        ch_h = self.ch_hidden[i][0]
        eks, sks = self.embed_ks, self.spade_ks

        b = x.size()[0]
        x = self.reshape_embed_input(x)
        
        # weights for the label embedding network  
        embedding_weights = None
        if self.adap_embed:
            fc_e = getattr(self, 'fc_spade_e_'+str(i))(x).view(b, -1)
            embedding_weights = self.reshape_weight(fc_e, [ch_out, ch_in, eks, eks])

        # weights for the 3 layers in SPADE module: conv_0, conv_1, and shortcut
        fc_0 = getattr(self, 'fc_spade_0_'+str(i))(x).view(b, -1)
        fc_1 = getattr(self, 'fc_spade_1_'+str(i))(x).view(b, -1)
        fc_s = getattr(self, 'fc_spade_s_'+str(i))(x).view(b, -1)
        weight_0 = self.reshape_weight(fc_0, [[ch_out, ch_h, sks, sks]]*2)
        weight_1 = self.reshape_weight(fc_1, [[ch_in, ch_h, sks, sks]]*2)
        weight_s = self.reshape_weight(fc_s, [[ch_out, ch_h, sks, sks]]*2)
        norm_weights = [weight_0, weight_1, weight_s]
        
        return embedding_weights, norm_weights

# embed target label
class LabelEmbedder(BaseNetwork):
    def __init__(self, opt, input_nc, netS=None, params_free_layers=0, first_layer_free=False):
        super().__init__()        
        self.opt = opt        
        activation = nn.LeakyReLU(0.2, True)        
        nf = opt.ngf
        nf_max = 1024
        self.netS = netS if netS is not None else opt.netS
        self.n_downsample_S = n_downsample_S = opt.n_downsample_G        
        self.params_free_layers = params_free_layers if params_free_layers != -1 else n_downsample_S
        self.first_layer_free = first_layer_free    
        ch = [min(nf_max, nf * (2 ** i)) for i in range(n_downsample_S + 1)]
       
        if not first_layer_free:
            layer = [nn.Conv2d(input_nc, nf, kernel_size=3, padding=1), activation]
            self.conv_first = nn.Sequential(*layer)
        
        # downsample
        for i in range(n_downsample_S):            
            layer = [nn.Conv2d(ch[i], ch[i+1], kernel_size=3, stride=2, padding=1), activation]
            if i >= params_free_layers or 'decoder' in netS:
                setattr(self, 'down_%d' % i, nn.Sequential(*layer))
        
        # upsample
        if 'decoder' in self.netS:
            for i in reversed(range(n_downsample_S)):                
                layer = [nn.ConvTranspose2d(ch[i+1], ch[i], kernel_size=3, stride=2, padding=1, output_padding=1), activation]                    
                if i >= params_free_layers:
                    setattr(self, 'up_%d' % i, nn.Sequential(*layer))            

    def forward(self, input, weights=None):
        if input is None: return None
        
        if self.first_layer_free:
            output = [batch_conv(input, weights[0])]
            weights = weights[1:]
        else:
            output = [self.conv_first(input)]
        for i in range(self.n_downsample_S):
            if i >= self.params_free_layers or 'decoder' in self.netS:                
                conv = getattr(self, 'down_%d' % i)(output[-1])
            else:                
                conv = batch_conv(output[-1], weights[i], stride=2)
            output.append(conv)

        if self.netS == 'encoder':            
            return output

        output = [output[-1]]        
        for i in reversed(range(self.n_downsample_S)):
            if i >= self.params_free_layers:                
                conv = getattr(self, 'up_%d' % i)(output[-1])            
            else:                
                conv = batch_conv(output[-1], weights[i], stride=0.5)
            output.append(conv)
        return output[::-1]