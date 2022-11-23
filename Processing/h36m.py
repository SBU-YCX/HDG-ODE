"""
    Name    : h36m.py (Human3.6M Dataset) 
    Author  : Yucheng Xing
"""
import os
import h5py
import torch
import cdflib
import numpy as np
import pandas as pd
import scipy.io as sio
from torch.utils.data import Dataset
from random import shuffle


def fetch_cdf(file_path_2d, file_path_3d, fps):
	## 2D File
	x2ds = cdflib.CDF(file_path_2d)[0][0]
	x2ds = (x2ds - x2ds.mean()) / x2ds.std()
	## 3D File
	x3ds = cdflib.CDF(file_path_3d)[0][0]
	x3ds = (x3ds - x3ds.mean()) / x3ds.std()
	## times
	ts = np.arange(0, x2ds.shape[0], 1, dtype=np.float32) * 1./fps
	return x2ds.reshape(ts.shape[0], -1, 2), x3ds.reshape(ts.shape[0], -1, 3), ts


def split_data(file_path="../Data/Human36M/raw/", save_path="../Data/Human36M/processed", seg_len=50, fps=50, pt=0.9, ps=0.7, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
	with h5py.File(os.path.join(save_path, 'data_{}_t{}_s{}.hdf5').format(seg_len, int(pt * 10), int(ps * 10)), 'w') as Dataset:
		## Selected Indices:
		joint_idx = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
		# ji_dict = [0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11, 12, 13, 14, 15, 16]
		joint_idx_dict = {ji : k for k, ji in enumerate(joint_idx)}
		## Static Adj Mat
		connection = np.array([
						[15, 14], [14, 13], 
						[13, 17], [17, 18], [18, 19], 
						[13, 25], [25, 26], [26, 27], 
						[13, 12], [12, 0], 
						[0, 1], [1, 2], [2, 3], 
						[0, 6], [6, 7], [7, 8]
					 ])
		static_adj = np.zeros((17, 17))
		for e in connection:
			static_adj[joint_idx_dict[e[0]]][joint_idx_dict[e[1]]] = 1.0
		static_adj = static_adj + static_adj.T
		Dataset.create_dataset('static_adj', data=static_adj)
		## Hierarchical Group
		group = [
					np.array([8]), 
					np.array([8, 2, 5, 9, 12, 15, 1]), 
					np.array([[2, 3, 4], [5, 6, 7], [9, 10, 11], [12, 13, 14], [15, 16, 17]])
				]
		for g in range(3):
			Dataset.create_dataset('group_{}'.format(g + 1), data=group[g])
		## Hierarchical Static Adj Mat
		static_adj_lv2 = np.zeros((7, 7))
		for e in connection:
			if joint_idx_dict[e[0]] + 1 in group[1] and joint_idx_dict[e[1]] + 1 in group[1]:
				static_adj_lv2[np.where(group[1] == joint_idx_dict[e[0]] + 1)[0].item()][np.where(group[1] == joint_idx_dict[e[1]] + 1)[0].item()] = 1.0
		static_adj_lv2 = static_adj_lv2 + static_adj_lv2.T
		Dataset.create_dataset('static_adj_lv2', data=static_adj_lv2)
		static_adj_lv3 = np.zeros((5, 3, 3))
		for sg in range(5):
			for e in connection:
				if joint_idx_dict[e[0]] + 1 in group[2][sg] and joint_idx_dict[e[1]] + 1 in group[2][sg]:
					static_adj_lv3[sg][np.where(group[2][sg] == joint_idx_dict[e[0]] + 1)[0].item()][np.where(group[2][sg] == joint_idx_dict[e[1]] + 1)[0].item()] = 1.0
			static_adj_lv3[sg, :, :] = static_adj_lv3[sg, :, :] + static_adj_lv3[sg, :, :].T
		Dataset.create_dataset('static_adj_lv3', data=static_adj_lv3)
		data_idx = 0
		data_idx_lst = []
		## Read Files
		for fi in [1, 5, 6, 7, 8, 9, 11]:
			for ai in os.listdir(os.path.join(file_path, '3D/S{}/MyPoseFeatures/D3_Positions/').format(fi)):
				ai = ai.split('.')[0]
			#for ai in ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'Walking', 'WalkingDog', 'WalkTogether']:
				file_2d = os.path.join(file_path, '2D/S{}/MyPoseFeatures/D2_Positions/{}.54138969.cdf').format(fi, ai)
				file_3d = os.path.join(file_path, '3D/S{}/MyPoseFeatures/D3_Positions/{}.cdf').format(fi, ai)
				x2ds, x3ds, ts = fetch_cdf(file_2d, file_3d, fps)
				start = 0
				while start + seg_len <= ts.shape[0]:
					end = start + seg_len
					## 2D & 3D Coordinates
					coord_2d = np.zeros((seg_len, 1, 17, 2))
					coord_2d[:, 0, :, :] = x2ds[start : end, joint_idx, :]
					coord_3d = np.zeros((seg_len, 1, 17, 3))
					coord_3d[:, 0, :, :] = x3ds[start : end, joint_idx, :]
					Dataset.create_dataset('x2ds_{}'.format(data_idx), data=coord_2d)
					Dataset.create_dataset('x3ds_{}'.format(data_idx), data=coord_3d)
					## Mask
					temporal_mask = np.zeros((seg_len, 1, 1, 1))
					t_obs_idx = np.random.choice(np.arange(0, seg_len), size=min(int(seg_len * pt), seg_len - 1), replace=False)
					temporal_mask[t_obs_idx] = 1.0
					spatial_mask = np.zeros((seg_len, 1, 17, 1))
					for dt in range(seg_len):
						s_obs_idx = np.random.choice(np.arange(0, 17), size=int(17 * ps), replace=False)
						spatial_mask[dt, :, s_obs_idx, :] = 1.0
					mask = spatial_mask * temporal_mask
					Dataset.create_dataset('masks_{}'.format(data_idx), data=mask)
					## Flatten & Hierarchical Dynamic Adj Mat
					dynamic_adj = np.zeros((seg_len, 1, 17, 17))
					dynamic_adj_lv1 = np.zeros((seg_len, 1, 1))
					dynamic_adj_lv2 = np.zeros((seg_len, 1, 7, 7))
					dynamic_adj_lv3 = np.zeros((seg_len, 1, 5, 3, 3))
					for dt in range(seg_len):
						# Lv1

						# Lv2
						dm = mask[dt, :, group[1] - 1].reshape((1, -1))
						dynamic_adj_lv2[dt] = static_adj_lv2 * np.matmul(dm.T, dm)
						# Lv3
						for sg in range(5):
							dm = mask[dt, :, group[2][sg] - 1].reshape((1, -1))
							dynamic_adj_lv3[dt, :, sg, :] = static_adj_lv3[sg] * np.matmul(dm.T, dm)
						# Flatten
						dm = mask[dt].reshape((1, -1))
						dynamic_adj[dt, :] = static_adj * np.matmul(dm.T, dm)
					Dataset.create_dataset('dynamic_adj_lv1_{}'.format(data_idx), data=dynamic_adj_lv1)
					Dataset.create_dataset('dynamic_adj_lv2_{}'.format(data_idx), data=dynamic_adj_lv2)
					Dataset.create_dataset('dynamic_adj_lv3_{}'.format(data_idx), data=dynamic_adj_lv3)
					Dataset.create_dataset('dynamic_adj_{}'.format(data_idx), data=dynamic_adj)
					## T
					Dataset.create_dataset('ts_{}'.format(data_idx), data=ts[start : end])
					start += seg_len
					data_idx_lst.append(data_idx)
					data_idx += 1
				print(file_3d, ' Done!')
		shuffle(data_idx_lst)
		train_lst = data_idx_lst[:int(train_ratio * data_idx)]
		valid_lst = data_idx_lst[int(train_ratio * data_idx):int((train_ratio + valid_ratio) * data_idx)]
		test_lst = data_idx_lst[int((train_ratio + valid_ratio) * data_idx):]
		Dataset.create_dataset('train_idx', data=np.array(train_lst, dtype=np.int32))
		Dataset.create_dataset('valid_idx', data=np.array(valid_lst, dtype=np.int32))
		Dataset.create_dataset('test_idx', data=np.array(test_lst, dtype=np.int32))
	return


class Human36M(Dataset): 
	def __init__(self, phase, method, data_path="../Data/Human36M/processed/data_50_t9_s7.hdf5"):
		super(Human36M, self).__init__()
		assert phase in ['train', 'valid', 'test'], "phase must be in ['train', 'valid', 'test']."
		assert method in ['single'], "method must be in ['single']."
		self.data = h5py.File(data_path, 'r')
		self.indices = self.data['{}_idx'.format(phase)]
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
		t = self.data['ts_{}'.format(index)][()].reshape((-1, 1))
		x2d = self.data['x2ds_{}'.format(index)][:]
		x3d = self.data['x3ds_{}'.format(index)][:]
		m = self.data['masks_{}'.format(index)][:]
		d_adj = self.data['dynamic_adj_{}'.format(index)][:, 0, :, :]
		d_adj_h1 = self.data['dynamic_adj_lv1_{}'.format(index)][:]
		d_adj_h2 = self.data['dynamic_adj_lv2_{}'.format(index)][:]
		d_adj_h3 = self.data['dynamic_adj_lv3_{}'.format(index)][:]
		kp = 1
		return {'t' : t.astype(np.float32), 
				'x2d' : x2d.astype(np.float32), 
				'x3d' : x3d.astype(np.float32), 
				'mask' : m.astype(np.float32), 
				's_adj_f' : s_adj.astype(np.float32), 
				's_adj_h' : [d_adj_h1[0].astype(np.float32), s_adj_h2.astype(np.float32), s_adj_h3.astype(np.float32)], 
				'd_adj_f' : d_adj.astype(np.float32), 
				'd_adj_h' : [d_adj_h1.astype(np.float32), d_adj_h2.astype(np.float32), d_adj_h3.astype(np.float32)], 
				'group' : [g1, g2, g3], 
				'num_p' : kp
			   }


'''
split_data(seg_len=100, pt=1.0, ps=0.6)
split_data(seg_len=100, pt=0.8, ps=0.5)
split_data(seg_len=100, pt=0.6, ps=0.4)
split_data(seg_len=100, ps=0.6, pt=0.8)
split_data(seg_len=100, ps=0.6, pt=0.6)
'''