import os
import pdb

import torch.nn as nn
import torch
import torch.nn.functional as F


class A2L(nn.Module):
    def __init__(self, opt, numFilters=64, filterWidth=21):
        super(A2L, self).__init__()
        self.opt = opt
        self.numFilters = numFilters
        self.filterWidth = filterWidth

        # self.audio_linear = nn.Linear(7, 128)
        self.conv = nn.Sequential(
            nn.Conv1d(7, self.numFilters, self.filterWidth,
                      stride=2, padding=0),
            nn.BatchNorm1d(self.numFilters),
            nn.LeakyReLU(0.3),
            nn.Dropout(opt.drop),
            nn.Conv1d(self.numFilters, 2*self.numFilters,
                      self.filterWidth, stride=2, padding=0),
            nn.BatchNorm1d(2*self.numFilters),
            nn.LeakyReLU(0.3),
            nn.Dropout(opt.drop),
            nn.Conv1d(2*self.numFilters, 4*self.numFilters,
                      self.filterWidth, stride=2, padding=0),
            nn.BatchNorm1d(self.numFilters*4),
            nn.LeakyReLU(0.3),
            nn.Dropout(opt.drop),
        )

    def forward(self, x):
        feat = x.transpose(1,2)
        feat = self.conv(feat)
        feat = feat.mean(-1)
        return feat

class SingleEmb(nn.Module):
    def __init__(self, opt):
        super(SingleEmb, self).__init__()
        self.opt = opt
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1),
            nn.ReLU(True),
            # nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1),
        )
        self.audio_l = nn.Sequential(
            nn.Conv1d(30528, 512, 3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.3),
            nn.Dropout(opt.drop),
            nn.Conv1d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.3),
        )

    def forward(self, x):
        # pdb.set_trace()
        b, t, c, w, h = x.size()
        feat = self.conv(x.reshape(b*t, c, w, h))
        feat = feat.view(feat.size(0), -1)
        feat = feat.reshape(b, t, -1)
        feat = feat.transpose(1,2)
        feat = self.audio_l(feat)
        feat = feat.transpose(1,2)
        return feat

class OneEmb(nn.Module):
    def __init__(self, opt):
        super(OneEmb, self).__init__()
        self.opt = opt
        self.audio_l = nn.Sequential(
            nn.Linear(7, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.3),
            nn.Dropout(opt.drop),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.3),
        )

    def forward(self, x):
        # pdb.set_trace()
        b, t = x.size()[:2]
        feat = self.audio_l(x.reshape(-1, x.size(-1)))
        feat = feat.reshape(b ,t, feat.size(-1))
        return feat