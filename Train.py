"""
    Name    : Train.py 
    Author  : Yucheng Xing
"""
import os
import argparse
import logging
import torch
import numpy as np
import scipy.io as sio
import torch.optim as optim
from Processing import MuPoTS, Human36M
from torch.utils.data import DataLoader
from Models import ODERNN, GCODERNN, DGCODERNN, HNODE, HGCODE, HDGODE, GCGRU, DGCGRU, STGCN, SemGCGRU, SemGCODERNN
from Losses import MaskedMSE, MPJPE, PCKh, ParamCount
from tqdm.auto import tqdm
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchinfo import summary


parser = argparse.ArgumentParser(description="Experimental Configuration.")
# Hyper-parameter
parser.add_argument('--dataset', 
					required=False, 
					default='mupots_single', 
					type=str, 
					help='Dataset')
parser.add_argument('--data_path', 
					required=False, 
					default='../Data', 
					type=str, 
					help='Data path.')
parser.add_argument('--model', 
					required=False, 
					default='ODERNN', 
					type=str, 
					help='Model name.')
parser.add_argument('--model_path', 
					required=False, 
					default='../Model', 
					type=str, 
					help='Trained model path.')
parser.add_argument('--pretrained', 
					required=False, 
					default=False, 
					type=bool, 
					help='Continue training or not.')
parser.add_argument('--log_path', 
					required=False, 
					default='../Log', 
					type=str, 
					help='Log file path.')
parser.add_argument('--batch_size', 
					required=False,
					default=12, 
					type=int, 
					help='Batch size.'
					)
parser.add_argument('--learning_rate', 
					required=False,
					default=0.01, 
					type=float, 
					help='Learning rate.'
					)
parser.add_argument('--max_epoch', 
					required=False, 
					default=100, 
					type=int, 
					help='Max number of epoches.')
parser.add_argument('--early_stop', 
					required=False,
					default=10, 
					type=int, 
					help='Early stop threshold.')
parser.add_argument('--dim_in', 
					required=False,
					default=2,
					type=int, 
					help='Dimension of input.')
parser.add_argument('--dim_out', 
					required=False,
					default=3,
					type=int, 
					help='Dimension of output.')
parser.add_argument('--dim_hidden', 
					required=False,
					default=16, 
					type=int, 
					help='Dimension of hidden features.')
parser.add_argument('--num_hidden', 
					required=False,
					default=[1, 2, 3], 
					type=list, 
					help='Number of layers in the network')
parser.add_argument('--dim_rnn', 
					required=False,
					default=6, 
					type=int, 
					help='Dimension of rnn features.')
parser.add_argument('--num_rnn', 
					required=False,
					default=[1, 2, 3], 
					type=list, 
					help='Number of blocks in the rnn network')
parser.add_argument('--kernel_size', 
					required=False,
					default=6, 
					type=int, 
					help='Size of 1D Conv kernel.')
parser.add_argument('--num_joint',
					required=False, 
					default=[16, 1, 6, 3], 
					type=list, 
					help='Number of joints [total & each level].')
parser.add_argument('--max_people',
					required=False, 
					default=3,
					type=int, 
					help='Max number of people.')
parser.add_argument('--seg_len',
					required=False, 
					default=50, 
					type=int, 
					help='Segment length.')
parser.add_argument('--delta_t',
					required=False, 
					default=1/120, 
					type=float, 
					help='delta t.')
parser.add_argument('--pt',
					required=False, 
					default=0.9, 
					type=float, 
					help='Temporal Un-Occlusion Ratio.')
parser.add_argument('--ps',
					required=False, 
					default=0.7, 
					type=float, 
					help='Spatial Un-Occlusion Ratio.')


def get_logger(args):
	''' Get Logger '''
	logging.basicConfig(level=logging.DEBUG)
	logger = logging.getLogger()
	if args.log_path:
		log_handler = logging.FileHandler(os.path.join(args.log_path, args.dataset, '{}.log').format(args.model))
		logger.addHandler(log_handler)
	return logger


def get_dataset(args):
	''' Get Dataset ''' 
	if args.dataset.lower() in ['mupots_single', 'mupots_multi']:
		train_set = MuPoTS(phase='train', 
						   method=args.dataset.lower().split('_')[1])
		valid_set = MuPoTS(phase='valid', 
						   method=args.dataset.lower().split('_')[1])
		test_set = MuPoTS(phase='test', 
						   method=args.dataset.lower().split('_')[1])
	elif args.dataset.lower() in ['h36m_single']:
		train_set = Human36M(phase='train', 
						   	method=args.dataset.lower().split('_')[1], 
						   	data_path="../Data/Human36M/processed/data_100_t{}_s{}.hdf5".format(int(args.pt * 10), int(args.ps * 10)))
		valid_set = Human36M(phase='valid', 
						   	method=args.dataset.lower().split('_')[1], 
						   	data_path="../Data/Human36M/processed/data_100_t{}_s{}.hdf5".format(int(args.pt * 10), int(args.ps * 10)))
		test_set = Human36M(phase='test', 
						   	method=args.dataset.lower().split('_')[1],
						   	data_path="../Data/Human36M/processed/data_100_t{}_s{}.hdf5".format(int(args.pt * 10), int(args.ps * 10)))
	else:
		train_set = None
		valid_set = None
		test_set = None
	train = DataLoader(dataset=train_set, 
					   shuffle=True, 
					   batch_size=args.batch_size, 
					   pin_memory=True)
	valid = DataLoader(dataset=valid_set, 
					   shuffle=True, 
					   batch_size=args.batch_size, 
					   pin_memory=True)
	test = DataLoader(dataset=test_set, 
					   shuffle=False, 
					   batch_size=args.batch_size, 
					   pin_memory=True)
	return train, valid, test
		

def get_model(args):
	''' Get Model '''
	if args.model.lower() == 'odernn':
		if args.dataset.split('_')[1].lower() == 'single':
			model = ODERNN(args.dim_in, 
					       args.dim_out, 
					       args.dim_rnn,
					       args.dim_hidden, 
					       args.num_hidden[-1], 
						   args.num_rnn[-1], 
					       args.num_joint[0],
					       args.delta_t)
		else:
			model = ODERNN(args.dim_in, 
					       args.dim_out, 
					       args.dim_rnn,
					       args.dim_hidden, 
					       args.num_hidden[-1], 
						   args.num_rnn[-1], 
					       args.num_joint[0] * 3,
					       args.delta_t)
	elif args.model.lower() == 'gcodernn':
		if args.dataset.split('_')[1].lower() == 'single':
			model = GCODERNN(args.dim_in, 
						 	 args.dim_out, 
					         args.dim_rnn,
						 	 args.dim_hidden, 
						 	 args.num_hidden[-1], 
						   	 args.num_rnn[-1], 
						 	 args.num_joint[0], 
						 	 args.delta_t, 
						 	 choice='F')
		else:
			model = GCODERNN(args.dim_in, 
							 args.dim_out, 
							 args.dim_rnn,
							 args.dim_hidden, 
							 args.num_hidden[-1], 
						   	 args.num_rnn[-1], 
							 args.num_joint[0] * 3, 
							 args.delta_t, 
							 choice='F')
	elif args.model.lower() == 'dgcodernn':
		if args.dataset.split('_')[1].lower() == 'single':
			model = DGCODERNN(args.dim_in, 
						  	  args.dim_out, 
						  	  args.dim_rnn,
						  	  args.dim_hidden, 
						  	  args.num_hidden[-1], 
						   	  args.num_rnn[-1], 
						  	  args.num_joint[0], 
						  	  args.delta_t, 
						  	  choice='F')
		else:
			model = DGCODERNN(args.dim_in, 
						  	  args.dim_out, 
						  	  args.dim_rnn,
						  	  args.dim_hidden, 
						  	  args.num_hidden[-1], 
						   	  args.num_rnn[-1], 
						  	  args.num_joint[0] * 3, 
						  	  args.delta_t, 
						  	  choice='F')
	elif args.model.lower() == 'semgcodernn':
		if args.dataset.split('_')[1].lower() == 'single':
			model = SemGCODERNN(args.dim_in, 
						  	  args.dim_out, 
						  	  args.dim_rnn,
						  	  args.dim_hidden, 
						  	  args.num_hidden[-1], 
						   	  args.num_rnn[-1], 
						  	  args.num_joint[0], 
						  	  args.delta_t, 
						  	  choice='F')
		else:
			model = SemGCODERNN(args.dim_in, 
						  	  args.dim_out, 
						  	  args.dim_rnn,
						  	  args.dim_hidden, 
						  	  args.num_hidden[-1], 
						   	  args.num_rnn[-1], 
						  	  args.num_joint[0] * 3, 
						  	  args.delta_t, 
						  	  choice='F')
	elif args.model.lower() == 'hnode':
		if args.dataset.split('_')[1].lower() == 'single':
			model = HNODE(args.dim_in, 
					  	  args.dim_out, 
					  	  args.dim_rnn,
					  	  args.dim_hidden, 
					  	  args.num_hidden, 
						  args.num_rnn, 
					  	  args.num_joint, 
					  	  args.delta_t)
		else:
			model = HNODE(args.dim_in, 
					  	  args.dim_out, 
					  	  args.dim_rnn,
					  	  args.dim_hidden, 
					  	  args.num_hidden, 
						  args.num_rnn, 
					  	  [args.num_joint[0], args.num_joint[1] * 3, args.num_joint[2] * 3, args.num_joint[3] * 3], 
					  	  args.delta_t)
	elif args.model.lower() == 'hgcode':
		model = HGCODE(args.dim_in, 
					   args.dim_out, 
					   args.dim_rnn,
					   args.dim_hidden, 
					   args.num_hidden, 
					   args.num_rnn, 
					   args.num_joint, 
					   args.delta_t, 
					   choice='F')
	elif args.model.lower() == 'hdgode':
		model = HDGODE(args.dim_in, 
					    args.dim_out, 
					    args.dim_rnn,
					    args.dim_hidden, 
					    args.num_hidden, 
						args.num_rnn, 
					    args.num_joint, 
					    args.delta_t, 
					    choice='F')
	elif args.model.lower() == 'gcgru':
		model = GCGRU(args.dim_in, 
					  args.dim_out, 
					  args.dim_rnn, 
					  args.num_rnn[-1], 
					  args.delta_t, 
				      choice='F')
	elif args.model.lower() == 'dgcgru':
		if args.dataset.split('_')[1].lower() == 'single':
			model = DGCGRU(args.dim_in, 
						   args.dim_out, 
						   args.dim_rnn, 
						   args.num_rnn[-1],
						   args.num_joint[0], 
						   args.delta_t, 
						   choice='F')
		else:
			model = DGCGRU(args.dim_in, 
						   args.dim_out, 
						   args.dim_rnn, 
						   args.num_rnn[-1],
						   args.num_joint[0] * 3, 
						   args.delta_t, 
						   choice='F')
	elif args.model.lower() == 'semgcgru':
		if args.dataset.split('_')[1].lower() == 'single':
			model = SemGCGRU(args.dim_in, 
						   args.dim_out, 
						   args.dim_rnn, 
						   args.num_rnn[-1],
						   args.num_joint[0], 
						   args.delta_t, 
						   choice='F')
		else:
			model = SemGCGRU(args.dim_in, 
						   args.dim_out, 
						   args.dim_rnn, 
						   args.num_rnn[-1],
						   args.num_joint[0] * 3, 
						   args.delta_t, 
						   choice='F')
	elif args.model.lower() == 'stgcn':
		if args.dataset.split('_')[1].lower() == 'single':
			model = STGCN(args.dim_in, 
					  	args.dim_out, 
					  	args.dim_hidden, 
					  	args.kernel_size, 
					  	args.num_rnn[-1], 
					  	args.num_joint[0])
		else:
			model = STGCN(args.dim_in, 
					  	args.dim_out, 
					  	args.dim_hidden, 
					  	args.kernel_size, 
					  	args.num_rnn[-1], 
					  	args.num_joint[0] * 3)
	return model.cuda()


def get_metric(x, y, m, k):
	''' Get Evaluation Metrics '''
	mse, mae, ms = 0., 0., 0
	for i in range(k.shape[0]):
		x_item = x[i, :, :k[i]]
		y_item = y[i, :, :k[i]]
		m_item = m[i, :, :k[i]]
		mse += ((x_item - y_item) ** 2).sum() / m_item.sum()
		mae += (torch.abs(x_item - y_item)).sum() / m_item.sum()
	return mse, mae


def get_test_metric(x, y, k):
	''' Get Testing Evaluation Metrics '''
	x = x[:, :, :k[0]]
	y = y[:, :, :k[0]]
	mpjpe, mpjpe_mean = MPJPE(x, y)
	pckh, pckh_mean = PCKh(x, y)
	pckh1, pckh_mean1 = PCKh(x, y, 0.1)
	pckh2, pckh_mean2 = PCKh(x, y, 1)
	return mpjpe, mpjpe_mean, pckh, pckh_mean, pckh_mean1, pckh_mean2


def train(args):
	''' Train '''
	# Get Logger
	logger = get_logger(args)
	print('Logger Done.')
	# Get Data
	train, valid, test = get_dataset(args)
	print('Dataset Done.')
	# Get Model
	model = get_model(args)
	if args.pretrained:
		model.load_state_dict(torch.load(os.path.join(args.model_path, args.dataset, '{}.pth').format(args.model)))
	print('Model Done.')
	# Get Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	print('Optimizer Done.')
	best_score = float('inf')
	bad_epoch = 0
	for epoch in range(1, args.max_epoch + 1):
		# Train
		model.train()
		pbar = tqdm(train)
		pbar.write('\x1b[1;35mTraining Epoch\t{:04d}:\x1b[0m'.format(epoch))
		for n, data in enumerate(pbar):
			optimizer.zero_grad()
			loss = model.get_loss(data)
			"""
			#if torch.isnan(loss):
				stat = model.state_dict()
				for k, v in stat.items():
					try:
						print(k, v.mean(), v.std(), v.max(), v.min())
					except:
						print(k, v)
			"""
			loss.backward()
			#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01, norm_type=2)
			torch.nn.utils.clip_grad_value_(model.parameters(), 100)
			optimizer.step()
			#torch.autograd.set_detect_anomaly(True)
			pbar.write('Epoch\t{:04d}, Iteration\t{:04d}: Loss\t{:6.4f}'.format(epoch, n, loss))
		# Valid
		model.eval()
		pbar = tqdm(valid)
		pbar.write('\x1b[1;35mValidation Epoch\t{:04d}:\x1b[0m'.format(epoch))
		mse_sum, mae_sum, num_item = 0., 0., 0.
		for n, data in enumerate(pbar):
			with torch.no_grad():
				ys = model.forward(data)[:, 1:]
				xs = data['x3d'][:, 1:].cuda()
				kp = data['num_p'].cuda()
				ms = torch.ones_like(xs).cuda()
				mse, mae = get_metric(xs, ys, ms, kp)
				mse_sum += mse
				mae_sum += mae
				num_item += xs.shape[0]
		rmse = torch.sqrt(mse_sum / num_item).detach().cpu().numpy()
		mae = (mae_sum / num_item).detach().cpu().numpy()
		pbar.write('\x1b[1;35mValidation Epoch\t{:04d}: RMSE - {:6.4f}\tMAE - {:6.4f}.\x1b[0m.\x1b[0m'.format(epoch, rmse, mae))
		logger.info('Epoch\t{:04d}: RMSE - {:6.4f}\tMAE - {:6.4f}'.format(epoch, rmse, mae))
		score = (rmse + mae) / 2
		# Update model
		if score < best_score:
			best_score = score
			bad_epoch = 1
			torch.save(model.state_dict(), os.path.join(args.model_path, args.dataset, '{}.pth').format(args.model))
		else:
			bad_epoch += 1
		# Early stop judjement
		if bad_epoch >= args.early_stop:
			#args.learning_rate *= 0.1
			#logger.info('Current lr: {:6.4f}'.format(args.learning_rate))
			#bad_epoch = 1
			break
		#if args.learning_rate < 5e-5:
		#	break
	# Close Logger
	for handler in logger.handlers[:]:
		logger.removeHandler(handler)
		handler.close()
	return


def test(args):
	''' Test '''
	# Get Logger
	logger = get_logger(args)
	print('Logger Done.')
	# Get Data
	_, _, test = get_dataset(args)
	print('Dataset Done.')
	# Get Model
	model = get_model(args)
	model.load_state_dict(torch.load(os.path.join(args.model_path, args.dataset, '{}.pth').format(args.model)))
	model.eval()
	print('Model Done.')
	# Test
	mse_sum, mae_sum, num_item = 0., 0., 0.
	mpjpe_mean_sum, pckh_mean_sum1, pckh_mean_sum2, pckh_mean_sum3 = 0., 0., 0., 0.
	time, flops = 0., 0.
	starter = torch.cuda.Event(enable_timing=True)
	ender = torch.cuda.Event(enable_timing=True)
	pbar = tqdm(test)
	pbar.write('\x1b[1;35mTestng:\x1b[0m')
	#print(parameter_count_table(model))
	for n, data in enumerate(pbar):
		with torch.no_grad():
			starter.record()
			ys = model.forward(data)[:, 1:]
			ender.record()
			#flops = FlopCountAnalysis(model, data)
			#print(flops.total())
			#print(summary(model, input_data=[data]))
			#return
			torch.cuda.synchronize()
			time += starter.elapsed_time(ender)
			xs = data['x3d'][:, 1:].cuda()
			kp = data['num_p'].cuda()
			ms = torch.ones_like(xs).cuda()
			mse, mae = get_metric(xs, ys, ms, kp)
			mse_sum += mse
			mae_sum += mae
			num_item += xs.shape[0]
			mpjpe, mpjpe_mean, pckh, pckh_mean1, pckh_mean2, pckh_mean3 = get_test_metric(xs, ys, kp)
			mpjpe_mean_sum += mpjpe_mean
			pckh_mean_sum1 += pckh_mean1
			pckh_mean_sum2 += pckh_mean2
			pckh_mean_sum3 += pckh_mean3
	rmse = torch.sqrt(mse_sum / num_item).detach().cpu().numpy()
	mae = (mae_sum / num_item).detach().cpu().numpy()
	mpjpe_mean_sum = (mpjpe_mean_sum / num_item).detach().cpu().numpy()
	pckh_mean_sum1 = (pckh_mean_sum1 / num_item).detach().cpu().numpy()
	pckh_mean_sum2 = (pckh_mean_sum2 / num_item).detach().cpu().numpy()
	pckh_mean_sum3 = (pckh_mean_sum3 / num_item).detach().cpu().numpy()
	time = (time / num_item)
	pc = ParamCount(model)
	#flop, param = profile(model, inputs = (torch.randn(1, 50, 3, 16, 2)))
	#print(flop, param)
	pbar.write('\x1b[1;35mTesting: RMSE - {:6.4f}\tMAE - {:6.4f}\tMPJPE - {:6.4f}\tPCKh@0.5\t - {:6.4f}\tPCKh@0.1\t - {:6.4f}\tPCKh@1.0\t - {:6.4f}\tParams Count\t - {:6.4f}\tTime - {:6.4f}.\x1b[0m.\x1b[0m'.format(rmse, mae, mpjpe_mean_sum, pckh_mean_sum1, pckh_mean_sum2, pckh_mean_sum3, pc, time))
	logger.info('Testing: RMSE - {:6.4f}\tMAE - {:6.4f}\tMPJPE - {:6.4f}\tPCKh@0.5\t - {:6.4f}\tPCKh@0.1\t - {:6.4f}\tPCKh@1.0\t - {:6.4f}\tParams Count\t - {:6.4f}\tTime - {:6.4f}'.format(rmse, mae, mpjpe_mean_sum, pckh_mean_sum1, pckh_mean_sum2, pckh_mean_sum3, pc, time))
	# Close Logger
	for handler in logger.handlers[:]:
		logger.removeHandler(handler)
		handler.close()
	return