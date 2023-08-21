"""
    Name    : mupots.py (MuPoTS-3D Dataset) 
    Author  : Yucheng Xing
"""
import os
import h5py
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
from torch.utils.data import Dataset
from random import shuffle


class MuPoTS(Dataset):
	def __init__(self, phase, method, data_path="../Data/MuPoTS/processed/data_50.hdf5"):
		super(MuPoTS, self).__init__()
		assert phase in ['train', 'valid', 'test'], "phase must be in ['train', 'valid', 'test']."
		assert method in ['multi', 'single'], "method must be in ['multi', 'single']."
		self.data = h5py.File(data_path, 'r')
		self.indices = self.data['{}_{}_idx'.format(phase, method)]
		self.size = self.indices.shape[0]
		self.phase = phase
		self.method = method
		return

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		s_adj = self.data['static_adj'][:]
		s_adj_h2 = self.data['static_adj_lv2'][:]
		s_adj_h3 = self.data['static_adj_lv3'][:]
		g1 = self.data['group_1'][:]
		g2 = self.data['group_2'][:]
		g3 = self.data['group_3'][:]
		index = self.indices[idx]
		t = self.data['ts_{}'.format(index[0])][()].reshape((-1, 1))
		if self.method == 'multi':
			x2d = (self.data['x2ds_{}'.format(index[0])][:])
			x3d = (self.data['x3ds_{}'.format(index[0])][:])
			m = self.data['masks_{}'.format(index[0])][:]
			m = np.expand_dims(m, axis=(3))
			d_adj = self.data['dynamic_adj_{}'.format(index[0])][:]
			d_adj_h1 = self.data['dynamic_adj_lv1_{}'.format(index[0])][:]
			d_adj_h2 = self.data['dynamic_adj_lv2_{}'.format(index[0])][:]
			d_adj_h3 = self.data['dynamic_adj_lv3_{}'.format(index[0])][:]
			kp = d_adj.shape[1]
			s_adj_f = np.zeros((3 * 16, 3 * 16))
			for pi in range(kp):
				s_adj_f[pi * 16 : (pi + 1) * 16, pi * 16 : (pi + 1) * 16] = s_adj
				for pj in range(kp):
					if d_adj_h1[0][pi][pj] == 1:
						s_adj_f[pi * 16 + 14, pj * 16 + 14] = 1
			d_adj_f = np.zeros((x2d.shape[0], 3 * 16, 3 * 16))
			for pi in range(kp):
				d_adj_f[:, pi * 16 : (pi + 1) * 16, pi * 16 : (pi + 1) * 16] = d_adj[:, pi]
				for pj in range(kp):
					d_adj_f[:, pi * 16 + 14, pj * 16 + 14] = d_adj_h1[:, pi, pj]
		else:
			x2d = self.data['x2ds_{}'.format(index[0])][:, index[1], :, :]
			x3d = self.data['x3ds_{}'.format(index[0])][:, index[1], :, :]
			m = self.data['masks_{}'.format(index[0])][:, index[1], :]
			x2d = np.expand_dims(x2d, axis=(1))
			x3d = np.expand_dims(x3d, axis=(1))
			m = np.expand_dims(m, axis=(1, 3))
			kp = 1
			s_adj_f = s_adj
			d_adj_f = self.data['dynamic_adj_{}'.format(index[0])][:, index[1], :, :]
			d_adj_h1 = self.data['dynamic_adj_lv1_{}'.format(index[0])][:, index[1], index[1]].reshape((-1, 1, 1))
			d_adj_h2 = self.data['dynamic_adj_lv2_{}'.format(index[0])][:, index[1], :, :]
			d_adj_h2 = np.expand_dims(d_adj_h2, axis=(1))
			d_adj_h3 = self.data['dynamic_adj_lv3_{}'.format(index[0])][:, index[1], :, :]
			d_adj_h3 = np.expand_dims(d_adj_h3, axis=(1))
		return {'t' : t.astype(np.float32), 
				'x2d' : x2d.astype(np.float32), 
				'x3d' : x3d.astype(np.float32), 
				'mask' : m.astype(np.float32), 
				's_adj_f' : s_adj_f.astype(np.float32), 
				's_adj_h' : [d_adj_h1[0].astype(np.float32), s_adj_h2.astype(np.float32), s_adj_h3.astype(np.float32)], 
				'd_adj_f' : d_adj_f.astype(np.float32), 
				'd_adj_h' : [d_adj_h1.astype(np.float32), d_adj_h2.astype(np.float32), d_adj_h3.astype(np.float32)], 
				'group' : [g1, g2, g3], 
				'num_p' : kp
				}
