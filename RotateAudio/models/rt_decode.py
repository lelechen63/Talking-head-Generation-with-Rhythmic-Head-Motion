import os
import pdb

import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTMDecode(nn.Module):
	def __init__(self, opt, numFilters=64, filterWidth=5):
		super(LSTMDecode, self).__init__()
		self.opt = opt
		self.lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True)
		if opt.rt_encode == 'tc':
			self.lstm2 = nn.LSTM(512, 256, num_layers=2, batch_first=True)

		if opt.rt_encode == 'tc_no_lstm':
			self.rt_linear = nn.Sequential(
				nn.Linear(512, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(True),
				nn.Linear(256, 6),
			)
		else:
			self.rt_linear = nn.Sequential(
				nn.Linear(256, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(True),
				nn.Linear(128, 6),
			)

	def forward(self, rt, audio, s_rt=None, max_len=0):
		pdb.set_trace()
		if s_rt is not None:
			s_rt = torch.mean(s_rt, dim=1, keepdim=True)
			res_list = []
			for i in range(audio.size(1)):
				if i == 0:
					out, hidden = self.lstm2(
						torch.cat([s_rt, audio[:, i:i+1]], -1))
				else:
					out, hidden = self.lstm2(
						torch.cat([out, audio[:, i:i+1]], -1), hidden)

				res_list.append(out)

			out = torch.cat(res_list, 1)
		else:
			rt = rt.unsqueeze(1).repeat(1, audio.size(1), 1)
			feat = torch.cat([rt, audio], -1)
			out, _ = self.lstm(feat)

		out = out.reshape(-1, out.size(-1))
		out = self.rt_linear(out)
		return out.reshape(audio.size(0), audio.size(1), -1)


class L2LDecode(nn.Module):
	def __init__(self, opt, numFilters=64, filterWidth=5):
		super(L2LDecode, self).__init__()
		self.opt = opt
		self.l = nn.Linear(512, 256*6+6)

	def forward(self, rt, audio):
		# pdb.set_trace()
		feat = self.l(rt)

		b, t = audio.size()[:2]
		w, _b = feat[:, :-6].view(b, 256, 6), feat[:, -6:]

		# all_out = []
		# for i in range(b):
		# 	out = F.linear(audio[i], w[i], _b[i])
		# 	all_out.append(out)
		# all_out = torch.stack(all_out)

		out = torch.bmm(audio, w) + _b.unsqueeze(1)
		

		return out


class RTDecode(nn.Module):
	def __init__(self, opt, numFilters=64, filterWidth=5):
		super(RTDecode, self).__init__()
		self.opt = opt
		self.lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True)
		self.rt_linear = nn.Sequential(
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Linear(256, 6),
		)

	def forward(self, rt, audio, s_rt=None, max_len=0):
		# pdb.set_trace()
		rt = rt.unsqueeze(1)
		res_list = []
		for i in range(audio.size(1)):
			if i == 0:
				out, hidden = self.lstm(rt)
			else:
				out, hidden = self.lstm(out, hidden)

			res_list.append(out)
		res_list = torch.cat(res_list, 1)

		res_list = res_list.reshape(-1, res_list.size(-1))
		res_list = self.rt_linear(res_list)
		return res_list.reshape(audio.size(0), audio.size(1), -1)


class LinearDecode(nn.Module):
	def __init__(self, opt):
		super(LinearDecode, self).__init__()
		self.opt = opt
		self.l_256 = nn.Sequential(
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
		)
		self.l = nn.Sequential(
			nn.Linear(512, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.Dropout(opt.drop),
			nn.Linear(128, 6),
		)

		self.lstm = nn.LSTM(256, 256)

	def forward(self, rt, s_rt, audio):
		# pdb.set_trace()
		# rt = rt.unsqueeze(1).repeat((1, audio.size(1), 1))
		# feat = torch.cat([rt, audio], -1)
		if len(audio.size()) == 3:
			audio = audio.squeeze(1)
		feat = torch.cat([rt, audio], -1)
		feat = self.l_256(feat)

		s_rt, _ = self.lstm(s_rt)
		feat = torch.cat([feat, s_rt[:, -1, :]], -1)

		out = self.l(feat).unsqueeze(1)
		return out


class LinearDecode_no_LSTM(nn.Module):
	def __init__(self, opt):
		super(LinearDecode_no_LSTM, self).__init__()
		self.opt = opt
		self.l1 = nn.Sequential(
			nn.Linear(768, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(True),
			nn.Dropout(opt.drop),
		)
		self.l2 = nn.Sequential(
			nn.Linear(1024, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(True),
			nn.Dropout(opt.drop),
		)
		self.l3 = nn.Sequential(
			nn.Linear(384, 64),
			nn.BatchNorm1d(64),
			nn.ReLU(True),
			nn.Dropout(opt.drop),
			nn.Linear(64, 6),
		)

	def forward(self, rt, audio):
		# pdb.set_trace()
		rt = rt.unsqueeze(1).repeat((1, audio.size(1), 1))
		feat = torch.cat([rt, audio], -1)
		b, t, d = feat.size()
		feat = feat.reshape(b*t, d)
		out1 = self.l1(feat)
		out2 = self.l2(torch.cat([out1, feat], -1))
		out3 = self.l3(torch.cat([out2, out1], -1))
		return out3.reshape(b, t, -1)

class DT_Discriminator(nn.Module):
	def __init__(self, opt):
		super(DT_Discriminator, self).__init__()
		self.opt = opt
		# self.l = nn.Sequential(
		# 	nn.Linear(63*3, 128),
		# 	nn.BatchNorm1d(128),
		# 	nn.ReLU(True),
		# 	nn.Dropout(opt.drop),
		# 	nn.Linear(128, 1),
		# )
		self.l = nn.Sequential(
			nn.Linear(24, 1),
		)
		self.loss = nn.BCELoss()

	def forward(self, seq, label):
		# seq = seq[:, 1:] - seq[:, :-1]
		# out = self.l(seq.view(seq.size(0), -1)).squeeze()

		# pdb.set_trace()
		seq_minus = seq[:, 1:] - seq[:, :-1]
		out = torch.cat([seq.mean(1), seq.std(1), seq_minus.mean(1), seq_minus.std(1)], -1)
		out = self.l(out).squeeze()
		out = torch.sigmoid(out)
		l = self.loss(out, label)

		return l