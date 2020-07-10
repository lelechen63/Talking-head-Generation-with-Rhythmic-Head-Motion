import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
import scipy.fftpack

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dataset import VoxLmark2rgbDataset
from options import get_parser
from models.rt_decode import DT_Discriminator
import utils


class PadSequence:
	def __call__(self, batch):
		sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

		sequences = [x[0] for x in sorted_batch]
		sequences_padded = torch.nn.utils.rnn.pad_sequence(
			sequences, batch_first=True)

		lengths = torch.LongTensor([len(x) for x in sequences])

		labels = [x[1] for x in sorted_batch]
		labels_padded = torch.nn.utils.rnn.pad_sequence(
			labels, batch_first=True)
		return sequences_padded, labels_padded, lengths

def G_train(rt, audio, rt_length, train=False):
	audio_feat = audio_encode(audio)

	rt_input, rt_gt = rt[:, :64], rt[:, 64:]
	audio_prev, audio_input = audio_feat[:, :64], audio_feat[:, 64:]
	if opt.rt_encode == 'tc_no_lstm' or opt.rt_encode == 'rt_lstm':
		rt_feat = rt_encode(rt_input, audio_prev)
		rt_output = decode(rt_feat, audio_input)
	else:
		rt_feat, rt_s_feat = rt_encode(rt_input, audio_prev)
		rt_output = decode(rt_feat, audio_input, s_rt=rt_s_feat)

	loss_mse = loss_func(rt_output, rt_gt)
	if train:
		optimizer1.zero_grad()
		loss_mse.backward()
		optimizer1.step()

	return loss_mse.item()

def D_train(rt, audio, rt_length, train=False):
	audio_feat = audio_encode(audio)

	rt_input, rt_gt = rt[:, :base_num], rt[:, base_num:]
	audio_prev, audio_input = audio_feat[:, :base_num], audio_feat[:, base_num:]

	d_real_loss = dis(rt_gt, torch.ones(rt_gt.size(0)).cuda())

	if opt.rt_encode == 'tc_no_lstm' or opt.rt_encode == 'rt_lstm':
		rt_feat = rt_encode(rt_input, audio_prev)
		rt_output = decode(rt_feat, audio_input)
	else:
		rt_feat, rt_s_feat = rt_encode(rt_input, audio_prev)
		rt_output = decode(rt_feat, audio_input, s_rt=rt_s_feat)

	d_fake_loss = dis(rt_output, torch.zeros(rt_gt.size(0)).cuda())
	if train:
		optimizer2.zero_grad()
		(d_real_loss+d_fake_loss).backward()
		optimizer2.step()

	return d_real_loss.item() + d_fake_loss.item()

if __name__ == '__main__':
	opt = get_parser()

	utils.init_log_dir(opt)

	train_set, val_set = VoxLmark2rgbDataset(
		opt, 'train'), VoxLmark2rgbDataset(opt, 'val')
	train_loader = DataLoader(
		train_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True, collate_fn=PadSequence())
	val_loader = DataLoader(
		val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=PadSequence())

	rt_encode, audio_encode, decode = utils.get_model(opt)
	rt_encode, audio_encode, decode = rt_encode.cuda(), audio_encode.cuda(), decode.cuda()
	dis = DT_Discriminator(opt)
	dis = dis.cuda()

	loss_func = nn.MSELoss()

	optimizer1 = optim.Adam([{'params': rt_encode.parameters()},
							{'params': audio_encode.parameters()},
							{'params': decode.parameters()}], opt.base_lr, weight_decay=opt.weight_decay)
	optimizer2 = optim.Adam([{'params': dis.parameters()}], opt.base_lr, weight_decay=opt.weight_decay)

	scheduler1 = utils.AdjustLR(
		optimizer1, [opt.base_lr, opt.base_lr, opt.base_lr], sleep_epochs=5, half=5, verbose=1)
	scheduler2 = utils.AdjustLR(
		optimizer2, [opt.base_lr], sleep_epochs=5, half=5, verbose=1)

	val_best_loss = 999.
	for epoch in range(opt.num_epoch):
		scheduler1.step(epoch)
		scheduler2.step(epoch)
		for step, pack in enumerate(train_loader):
			rt = pack[0].float()
			audio = pack[1].float().unsqueeze(2)
			rt_length = pack[2]
			
			predict_num = random.randint(16, 96)
			base_num = random.randint(32, 96)
			if min(rt_length) <= (base_num+predict_num):
				random_start_idx = 0
			else:
				random_start_idx = random.sample(
					range(min(rt_length)-base_num-predict_num), 1)[0]
			rt = rt[:, random_start_idx:random_start_idx+base_num+predict_num].cuda()
			audio = audio[:, random_start_idx:random_start_idx+base_num+predict_num].cuda()

			D_loss = D_train(rt, audio, rt_length, train=True)
			G_loss = G_train(rt, audio, rt_length, train=True)

			if (step+1) % 25 == 0:
				print('epoch:{} step:{}/{} train_g_loss:{:.4f} train_d_loss:{:.4f}'.format(epoch,
																	 step+1, len(train_loader), G_loss, D_loss), end=' ')
				torch.save({'encode': rt_encode.state_dict(),
							'mid_net': audio_encode.state_dict(),
							'decode': decode.state_dict()}, os.path.join('./save', opt.name, 'check', 'model_{}_{}.pth.tar'.format(epoch, step+1)))

				rt_encode, audio_encode, decode = rt_encode.eval(), audio_encode.eval(), decode.eval()
				with torch.no_grad():
					for step, pack in enumerate(val_loader):
						rt = pack[0].float()
						audio = pack[1].float().unsqueeze(2)
						rt_length = pack[2]
						
						rt = rt[:, :128].cuda()
						audio = audio[:, :128].cuda()

						predict_num = 64
						base_num = 64
						D_loss = D_train(rt, audio, rt_length)
						G_loss = G_train(rt, audio, rt_length)

						print('val_g_loss:{:.4f} val_d_loss:{:.4f}'.format(G_loss, D_loss), end=' ')

					if (D_loss+G_loss) < val_best_loss:
						print('best')
						val_best_loss = (D_loss+G_loss)
						torch.save({'encode': rt_encode.state_dict(),
									'mid_net': audio_encode.state_dict(),
									'decode': decode.state_dict()}, os.path.join('./save', opt.name, 'best_model.pth.tar'))
					else:
						print('')

				rt_encode, audio_encode, decode = rt_encode.train(), audio_encode.train(), decode.train()
