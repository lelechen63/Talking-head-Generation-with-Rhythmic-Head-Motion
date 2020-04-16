# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn as nn
import functools
import copy
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import actvn as actvn

import pdb

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opt, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 subarch='n_layers', num_D=3, getIntermFeat=False, stride=2, gpu_ids=[]):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.getIntermFeat = getIntermFeat
        self.subarch = subarch
     
        for i in range(num_D):            
            netD = self.create_singleD(opt, subarch, input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride)
            setattr(self, 'discriminator_%d' % i, netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def create_singleD(self, opt, subarch, input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride):
        if subarch == 'adaptive':
            netD = AdaptiveDiscriminator(opt, input_nc, ndf, n_layers, norm_layer, getIntermFeat, opt.adaptive_D_layers)        
        elif subarch == 'n_layers':
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride)
        else:
            raise ValueError('unrecognized discriminator sub architecture %s' % subarch)
        return netD

    # should return list of outputs
    def singleD_forward(self, model, input, ref):
        if self.subarch == 'adaptive':
            return model(input, ref)
        elif self.getIntermFeat:
            return model(input)
        else:
            return [model(input)]

    # should return list of list of outputs
    def forward(self, input, ref=None):        
        result = []
        input_downsampled = input
        ref_downsampled = ref
        for i in range(self.num_D):
            model = getattr(self, 'discriminator_%d' % i)
            result.append(self.singleD_forward(model, input_downsampled, ref_downsampled))            
            input_downsampled = self.downsample(input_downsampled)
            ref_downsampled = self.downsample(ref_downsampled) if ref is not None else None
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False, stride=2):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers        

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))        
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, False)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            item = [
                norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                nn.LeakyReLU(0.2, False)
            ]
            sequence += [item]
                
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, False)
        ]]
        
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        
        for n in range(len(sequence)):                        
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))        

    def forward(self, input):
        res = [input]
        for n in range(self.n_layers + 2):            
            model = getattr(self, 'model'+str(n))
            x = model(res[-1])
            res.append(x)
        if self.getIntermFeat:
            return res[1:]
        else:
            return res[-1]

class AdaptiveDiscriminator(BaseNetwork):
    def __init__(self, opt, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False, adaptive_layers=1):
        super(AdaptiveDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.adaptive_layers = adaptive_layers
        self.input_nc = input_nc
        self.ndf = ndf
        self.kw = kw = 4
        self.padw = padw = int(np.ceil((kw-1.0)/2))  
        self.actvn = actvn = nn.LeakyReLU(0.2, True)
                
        self.sw = opt.fineSize // 8
        self.sh = int(self.sw / opt.aspect_ratio)
        self.ch = self.sh * self.sw        

        nf = ndf        
        self.fc_0 = nn.Linear(self.ch, input_nc*(kw**2))
        self.encoder_0 = nn.Sequential(*[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), actvn])
        for n in range(1, self.adaptive_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)             
            setattr(self, 'fc_'+str(n), nn.Linear(self.ch, nf_prev*(kw**2)))            
            setattr(self, 'encoder_'+str(n), nn.Sequential(*[(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)), actvn]))

        sequence = []
        nf = ndf * (2**(self.adaptive_layers-1))
        for n in range(self.adaptive_layers, n_layers+1):
            nf_prev = nf
            nf = min(nf * 2, 512) 
            stride = 2 if n != n_layers else 1
            item = [norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)), actvn]
            sequence += [item]                
        
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            setattr(self, 'model'+str(n + self.adaptive_layers), nn.Sequential(*sequence[n]))

    def gen_conv_weights(self, encoded_ref):
        models = []
        b = encoded_ref[0].size()[0]
        nf = self.ndf
        actvn = self.actvn                
        weight = self.fc_0(nn.AdaptiveAvgPool2d((self.sh, self.sw))(encoded_ref[0]).view(b*nf, -1))        
        weight = weight.view(b, nf, self.input_nc, self.kw, self.kw)
        model0 = []
        for i in range(b):
            model0.append(self.ConvN(functools.partial(F.conv2d, weight=weight[i], stride=2, padding=self.padw), 
                nn.InstanceNorm2d(nf), actvn))
        
        models.append(model0)
        for n in range(1, self.adaptive_layers):            
            ch = encoded_ref[n].size()[1]
            x = nn.AdaptiveAvgPool2d((self.sh, self.sw))(encoded_ref[n]).view(b*ch, -1)
            weight = getattr(self, 'fc_'+str(n))(x)
            
            nf_prev = nf
            nf = min(nf * 2, 512) 
            weight = weight.view(b, nf, nf_prev, self.kw, self.kw)
            model = []
            for i in range(b):
                model.append(self.ConvN(functools.partial(F.conv2d, weight=weight[i], stride=2, padding=self.padw), 
                    nn.InstanceNorm2d(nf), actvn))
            
            models.append(model)
        return models

    class ConvN(nn.Module):
        def __init__(self, conv, norm, actvn):
            super().__init__()                
            self.conv = conv
            self.norm = norm
            self.actvn = actvn

        def forward(self, x):
            x = self.conv(x)            
            out = self.norm(x)
            out = self.actvn(out)
            return out

    def encode(self, ref):
        encoded_ref = [ref]
        for n in range(self.adaptive_layers):
            encoded_ref.append(getattr(self, 'encoder_'+str(n))(encoded_ref[-1]))        
        return encoded_ref[1:]

    def batch_conv(self, conv, x):        
        y = conv[0](x[0:1])
        for i in range(1, x.size()[0]):
            yi = conv[i](x[i:i+1])
            y = torch.cat([y, yi])
        return y

    def forward(self, input, ref):
        encoded_ref = self.encode(ref)
        models = self.gen_conv_weights(encoded_ref)
        res = [input]
        for n in range(self.n_layers+2):            
            if n < self.adaptive_layers:
                res.append(self.batch_conv(models[n], res[-1]))
            else:                
                res.append(getattr(self, 'model'+str(n))(res[-1]))        
        if self.getIntermFeat:
            return res[1:]
        else:
            return res[-1]

class SyncDiscriminator(BaseNetwork):
    def __init__(self, opt, img_size=256, ndf=64, nf_final=256, n_layers=5, tot_layers=5, \
        kw=4, stride=2, audioW=21):
        super(SyncDiscriminator, self).__init__()
        assert n_layers <= tot_layers
        # define structure
        img_final_size = img_size
        audio_final_size = 1920
        padw = self.calculate_padding(kw, stride, img_size) // 2
        img_encoder = [[nn.Conv2d(3, ndf, kernel_size=kw, stride=stride, padding=padw),
                        nn.BatchNorm2d(ndf),
                        nn.LeakyReLU(0.3, False)]]
        audio_encoder = [[nn.Conv1d(1, ndf, kernel_size=audioW, stride=2, padding=0),
                        nn.BatchNorm1d(ndf),
                        nn.LeakyReLU(0.3, False)]]
        img_final_size = (img_final_size - kw + 2 * padw) // stride + 1
        audio_final_size = (audio_final_size - audioW) // 2 + 1

        # downsample
        nf = ndf
        for i in range(n_layers-1):
            nf_pref = nf
            nf = min(2 * nf, 256)
            padw = self.calculate_padding(kw, stride, img_final_size) // 2
            img_encoder.append([nn.Conv2d(nf_pref, nf, kernel_size=kw, stride=stride, padding=padw),
                        nn.BatchNorm2d(nf),
                        nn.LeakyReLU(0.3, False)])
            img_final_size = (img_final_size - kw + 2 * padw) // stride + 1
        # conv
        for i in range(tot_layers-n_layers):
            nf_pref = nf
            nf = min(2 * nf, 256)
            padw = self.calculate_padding(kw, 1, img_final_size) // 2
            img_encoder.append([nn.Conv2d(nf_pref, nf, kernel_size=kw, stride=1, padding=padw),
                        nn.BatchNorm2d(nf),
                        nn.LeakyReLU(0.3, False)])
            img_final_size = (img_final_size - kw + 2 * padw) // 1 + 1
        # audio (keep down sample)
        nf = ndf
        for i in range(tot_layers-1):
            nf_pref = nf
            nf = min(2 * nf, 256)
            audio_encoder.append([nn.Conv1d(nf_pref, nf, kernel_size=audioW, stride=2, padding=0, dilation=1),
                        nn.BatchNorm1d(nf),
                        nn.LeakyReLU(0.3, False)])
            audio_final_size = (audio_final_size - audioW) // 2 + 1

        audio_encoder.append([nn.Conv1d(nf, nf_final, kernel_size=41, stride=1, padding=0), nn.Tanh()])

        for i in range(len(img_encoder)):
            setattr(self, 'img_encoder_'+str(i), nn.Sequential(*img_encoder[i]))
        for i in range(len(audio_encoder)):
            setattr(self, 'audio_encoder_'+str(i), nn.Sequential(*audio_encoder[i]))

        # self.slp = nn.Linear(nf_final*2, 1, bias=True)
        padw = self.calculate_padding(kw, 2, img_final_size) // 2
        self.comb_conv_1 = nn.Sequential(*[nn.Conv2d(nf_final*2, 512, kernel_size=kw, stride=2, padding=padw),
                                           nn.BatchNorm2d(nf_final*2),
                                           nn.LeakyReLU(0.3, False)])
        img_final_size = (img_final_size - kw + 2 * padw) // 2 + 1

        padw = self.calculate_padding(kw, 2, img_final_size) // 2
        self.comb_conv_2 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=kw, stride=2, padding=padw),
                                           nn.BatchNorm2d(nf_final*2),
                                           nn.LeakyReLU(0.3, False)])
        img_final_size = (img_final_size - kw + 2 * padw) // 2 + 1

        self.comb_conv_cls = nn.Conv2d(512, 1, kernel_size=img_final_size, stride=1, padding=0)
        self.loss_fun = nn.MSELoss()

        # other parameter
        self.img_layers = len(img_encoder)
        self.audio_layers = len(audio_encoder)

    def forward(self, img, audio, is_real):
        img_f = img
        audio_f = audio
        # encode
        for i in range(self.img_layers):
            img_f = getattr(self, 'img_encoder_'+str(i))(img_f)
        for i in range(self.audio_layers):
            audio_f = getattr(self, 'audio_encoder_'+str(i))(audio_f)

        # reshape audio
        audio_f = audio_f.unsqueeze(-1).repeat([1, 1, img_f.shape[2], img_f.shape[3]])
        fea = torch.cat([img_f, audio_f], axis=1)
        fea = self.comb_conv_1(fea)
        fea = self.comb_conv_2(fea)
        fea = self.comb_conv_cls(fea)
        fea = fea.view(fea.shape[0], -1)
        
        # loss
        target = torch.zeros_like(fea).cuda(fea.get_device())
        if is_real:
            target += 1
        loss = self.loss_fun(fea, target)

        return fea, loss

class FrameDiscriminator(BaseNetwork):
    def __init__(self, opt, img_size=256, ch=6, ndf=64, n_layers=6, kw=4, stride=2):
        super(FrameDiscriminator, self).__init__()
        # define structure
        img_final_size = img_size
        padw = self.calculate_padding(kw, stride, img_size) // 2
        discriminator = [[nn.Conv2d(ch, ndf, kernel_size=kw, stride=stride, padding=padw),
                        nn.BatchNorm2d(ndf),
                        nn.LeakyReLU(0.3, False)]]
        img_final_size = (img_final_size - kw + 2 * padw) // stride + 1

        nf = ndf
        for i in range(n_layers-1):
            nf_prev = nf
            nf = min(512, 2 * nf)
            discriminator.append([nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw),
                        nn.BatchNorm2d(nf),
                        nn.LeakyReLU(0.3, False)])
            img_final_size = (img_final_size - kw + 2 * padw) // stride + 1

        discriminator.append([nn.Conv2d(nf, 1, kernel_size=img_final_size, stride=1, padding=0)])

        for i in range(len(discriminator)):
            setattr(self, 'layer_'+str(i), nn.Sequential(*discriminator[i]))

        self.loss_fun = nn.MSELoss()
        self.n_layers = len(discriminator)

    def forward(self, concat_img, is_real=True):
        fea = concat_img
        for i in range(self.n_layers):
            fea = getattr(self, 'layer_'+str(i))(fea)

        fea = fea.reshape(fea.shape[0], -1)
        target = torch.zeros_like(fea).cuda(fea.get_device())

        if is_real:
            target += 1
        loss = self.loss_fun(fea, target)
        return fea, loss

class SepDiscriminator(BaseNetwork):
    def __init__(self, opt, ndf, img_size=256, audio_size=1920):
        super(SepDiscriminator, self).__init__()
        self.ref_img_D = FrameDiscriminator(opt, ch=6, ndf=ndf)
        self.lmark_img_D = FrameDiscriminator(opt, ch=4, ndf=ndf)
        self.audio_img_D = SyncDiscriminator(opt, ndf=ndf, n_layers=4)

    def forward(self, img, ref_img, lmark, audio, is_real=True):
        ref_pred, lmark_pred, audio_pred = None, None ,None
        ref_loss, lmark_loss, audio_loss = torch.FloatTensor(1).fill_(0), \
                                           torch.FloatTensor(1).fill_(0), \
                                           torch.FloatTensor(1).fill_(0)

        if ref_img is not None:
            ref_pred, ref_loss = self.ref_img_D(torch.cat([img, ref_img], axis=1), is_real)
        if lmark is not None:
            lmark_pred, lmark_loss = self.lmark_img_D(torch.cat([img, lmark], axis=1), is_real)
        if audio is not None:
            audio_pred, audio_loss = self.audio_img_D(img, audio, is_real)

        preds = [ref_pred, lmark_pred, audio_pred]
        losses = [ref_loss, lmark_loss, audio_loss]

        return preds, losses

class SepFeaDiscriminator(BaseNetwork):
    def __init__(self, opt, ndf, n_layers, norm_layer, getIntermFeat, stride):
        super(SepFeaDiscriminator, self).__init__()
        self.ref_img_D = NLayerDiscriminator(input_nc=6, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer, getIntermFeat=getIntermFeat,\
                                             stride=stride)
        self.lmark_img_D = NLayerDiscriminator(input_nc=4, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer, getIntermFeat=getIntermFeat,\
                                             stride=stride)
        self.audio_img_D = AudioFeaDiscriminator(input_nc=4, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer, getIntermFeat=getIntermFeat,\
                                             stride=stride, nf_final=256)
    def forward(self, img, ref_img, lmark, audio):
        res = [None]*3
        if ref_img is not None:
            ref_res = self.ref_img_D(torch.cat([img, ref_img], axis=1))
            res[0] = ref_res
        if lmark is not None:
            lmark_res = self.lmark_img_D(torch.cat([img, lmark], axis=1))
            res[1] = lmark_res
        if audio is not None:
            audio_res = self.audio_img_D(img, audio)
            res[2] = audio_res
        return res

class AudioFeaDiscriminator(BaseNetwork):
    def __init__(self, input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride, nf_final=256, audio_encoder=None):
        super(AudioFeaDiscriminator, self).__init__()
        self.audio_img_D = NLayerDiscriminator(input_nc=4, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer, getIntermFeat=getIntermFeat,\
                                             stride=stride)
        self.audio_encoder = AudioEncoder(ndf, nf_final) if audio_encoder is None else audio_encoder

    def forward(self, img, audio):
        audio_fea = self.audio_encoder(audio)
        audio_res = self.audio_img_D(torch.cat([img, audio_fea], axis=1))
        
        return audio_res

    def set_audio_encoder(self, audio_encoder):
        self.audio_encoder = audio_encoder
    def get_audio_encoder(self):
        return self.audio_encoder

class AudioEncoder(BaseNetwork):
    def __init__(self, ndf, nf_final, audioW=21, tot_layers=5):
        super(AudioEncoder, self).__init__()
        # define structure
        audio_final_size = 1920
        audio_encoder = [[nn.Conv1d(1, ndf, kernel_size=audioW, stride=2, padding=0),
                        nn.BatchNorm1d(ndf),
                        nn.LeakyReLU(0.3, False)]]
        audio_final_size = (audio_final_size - audioW) // 2 + 1

        # audio (keep down sample)
        nf = ndf
        for i in range(tot_layers-1):
            nf_pref = nf
            nf = min(2 * nf, 256)
            audio_encoder.append([nn.Conv1d(nf_pref, nf, kernel_size=audioW, stride=2, padding=0, dilation=1),
                        nn.BatchNorm1d(nf),
                        nn.LeakyReLU(0.3, False)])
            audio_final_size = (audio_final_size - audioW) // 2 + 1

        audio_encoder.append([nn.Conv1d(nf, nf_final, kernel_size=audio_final_size, stride=1, padding=0), nn.Tanh()])

        for i in range(len(audio_encoder)):
            setattr(self, 'audio_encoder_'+str(i), nn.Sequential(*audio_encoder[i]))

        # other parameter
        self.audio_layers = len(audio_encoder)

    def forward(self, audio):
        audio_f = audio
        # encode
        for i in range(self.audio_layers):
            audio_f = getattr(self, 'audio_encoder_'+str(i))(audio_f)
        
        # reshape
        width = audio_f.shape[-2]
        audio_f = audio_f.unsqueeze(1).repeat([1, 1, 1, width])

        return audio_f