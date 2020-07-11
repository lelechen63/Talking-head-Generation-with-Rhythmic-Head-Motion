import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dataset import VoxLmark2rgbDataset
from options import get_parser
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


if __name__ == '__main__':
	opt = get_parser()

	train_set, test_set = VoxLmark2rgbDataset(
		opt, 'train'), VoxLmark2rgbDataset(opt, 'test')
	train_loader = DataLoader(
		train_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True, collate_fn=PadSequence())
	test_loader = DataLoader(
		test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=PadSequence())

	rt_encode, audio_encode, decode = utils.get_model(opt)
	checkpoint = torch.load('./save/{}/best_model.pth.tar'.format(opt.name), map_location=torch.device('cpu'))
	rt_encode.load_state_dict(checkpoint['encode'])
	audio_encode.load_state_dict(checkpoint['mid_net'])
	decode.load_state_dict(checkpoint['decode'])
	rt_encode, audio_encode, decode = rt_encode.cuda(), audio_encode.cuda(), decode.cuda()
	rt_encode.eval()
	audio_encode.eval()
	decode.eval()

	loss_func = nn.MSELoss()

	total_loss = []
	with torch.no_grad():
		for step, pack in enumerate(test_loader):
			if step == 100:
				break
			rt = pack[0].float()
			audio = pack[1].float().unsqueeze(2)
			rt_length = pack[2]
			
			rt = rt[:, :128].cuda()
			audio = audio[:, :128].cuda()
			audio_feat = audio_encode(audio)

			rt_input, rt_gt = rt[:, :64], rt[:, 64:128]
			audio_prev, audio_input = audio_feat[:, :64], audio_feat[:, 64:128]
			if opt.rt_encode == 'tc_no_lstm' or opt.rt_encode == 'rt_lstm':
				rt_feat = rt_encode(rt_input, audio_prev)
				rt_output = decode(rt_feat, audio_input)
			else:
				rt_feat, rt_s_feat = rt_encode(rt_input, audio_prev)
				rt_output = decode(rt_feat, audio_input, s_rt=rt_s_feat)

			loss = loss_func(rt_output, rt_gt)

			all_res = rt_output.cpu().numpy()
			rt_gt = rt_gt.cpu().numpy()
			total_loss.append(np.sum(loss.cpu().numpy()))

			fig, a =  plt.subplots(2, 3, figsize=(20, 10))
			for i in range(2):
				for j in range(3):
					a[i][j].plot(range(len(all_res[0, :, i*3+j])), all_res[0, :, i*3+j], label='pred')
					a[i][j].plot(range(len(rt_gt[0, :, i*3+j])), rt_gt[0, :, i*3+j], label='gt')

			plt.legend()
			plt.savefig('./save/{}/imgs/{}.png'.format(opt.name, step))
			plt.close()

	print('test mse loss: {}'.format(np.sum(total_loss)))