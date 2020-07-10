import os
import pdb

import torch.nn as nn
import torch
import torch.nn.functional as F

class RT_LSTM(nn.Module):
	def __init__(self, opt):
		super(RT_LSTM, self).__init__()
		self.opt = opt

		self.l = nn.Sequential(
			nn.Linear(6, 256),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(True),
			nn.Dropout(opt.drop),
			nn.Linear(256, 512),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(True),
			nn.Dropout(opt.drop),
		)

		self.lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True)

	def forward(self, feat):
		b, t = feat.size()[:2]
		s_feat = self.l(feat[:, :, :6].reshape(-1, 6))
		s_feat = s_feat.reshape(b, t, s_feat.size(-1))
		s_feat,_ = self.lstm(s_feat)

		return s_feat.mean(1)


class TemporalConv(nn.Module):
	def __init__(self, opt):
		super(TemporalConv, self).__init__()
		self.opt = opt
		self.numFilters = 64
		self.filterWidth = 5

		self.backend_conv1 = nn.Sequential(
				nn.Conv1d(13, self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1),
				nn.BatchNorm1d(self.numFilters),
				nn.LeakyReLU(0.3),
				nn.Dropout(opt.drop),
				nn.Conv1d(self.numFilters, 2*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1),
				nn.BatchNorm1d(self.numFilters*2),
				nn.LeakyReLU(0.3),
				nn.Dropout(opt.drop),
				nn.Conv1d(2*self.numFilters, 4*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1),
				nn.BatchNorm1d(self.numFilters*4),
				nn.LeakyReLU(0.3),
				nn.Dropout(opt.drop),
			)

		self.l = nn.Sequential(
			nn.Linear(6, 128),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(0.3),
			nn.Linear(128, 256),
		)

		self.lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True)

	def forward(self, feat):
		# pdb.set_trace()
		b, t, d = rt.size()
		rt = self.l(rt.reshape(b*t, d))
		rt = rt.reshape(b, t, -1)
		feat = torch.cat([rt, audio_feat], -1)

		return h, s_feat

class TemporalConv_no_LSTM(nn.Module):
	def __init__(self, opt):
		super(TemporalConv_no_LSTM, self).__init__()
		self.opt = opt
		self.hiddenDim = 64
		self.numFilters = 512
		self.filterWidth = 5

		self.l = nn.Sequential(
			nn.Linear(6, 128),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(0.3),
			nn.Linear(128, 256),
		)

		self.backend_conv1 = nn.Sequential(
				nn.Conv1d(512, self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1),
				nn.BatchNorm1d(self.numFilters),
				nn.LeakyReLU(0.3),
				nn.Dropout(opt.drop),
				nn.Conv1d(self.numFilters, self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1),
				nn.BatchNorm1d(self.numFilters),
				nn.LeakyReLU(0.3),
				nn.Dropout(opt.drop),
			)

	def forward(self, rt, audio_feat):
		# pdb.set_trace()
		b, t, d = rt.size()
		rt = self.l(rt.reshape(b*t, d))
		rt = rt.reshape(b, t, -1)
		feat = torch.cat([rt, audio_feat], -1)
		feat = feat.transpose(1,2)
		# feat = self.backend_conv1(feat)
		# feat = feat.mean(-1)

		h = self.backend_conv1(feat)

		return h.mean(-1)