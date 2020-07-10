import argparse

# 1 64-audio-lstm
# 2 rt-lstm + audio-single-prediction
# 3 64-audio-lstm no 1d audio encode
# 4 only 64-rt-lstm 2layer lstm
# 5 tc tc linear mse + dis
# 6 tc tc l2l mse + dis(std)
# 7 same above + random len
# 8 same above + smooth
# 9 same above + dis(std+12) 
# -- CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --rt_encode tc_no_lstm --audio_encode se --rt_decode l2l --batch_size 64 --drop 0.1 --name 9 --num_workers 4
# 10 same above + lstm

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default='lip')
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--num_workers', type=int, default=16)
	parser.add_argument('--dataroot', type=str, default="/home/cxu-serve/p1/common/voxceleb2") 
	parser.add_argument('--num_epoch', type=int, default=100)

	parser.add_argument('--rt_encode', type=str, default='tc')
	parser.add_argument('--audio_encode', type=str, default='a2l')
	parser.add_argument('--rt_decode', type=str, default='lstm')

	parser.add_argument('--drop', type=float, default=0)
	parser.add_argument('--base_lr', type=float, default=3e-4)
	parser.add_argument('--weight_decay', type=float, default=2e-5)

	opt = parser.parse_args()

	return opt
