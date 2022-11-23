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


def fetch_mat(file_path_1, file_path_2, fps):
	## Coordinate File
	data_mat = sio.loadmat(file_path_1)['annotations']
	x2ds, x3ds = [], []
	x2ds_sum, x2ds_sqr_sum, x3ds_sum, x3ds_sqr_sum, k = 0., 0., 0., 0., 0
	for t in range(data_mat.shape[0]):
		x2d, x3d = [], []
		for p in range(data_mat.shape[1]):
			data = data_mat[t][p]
			p2d = data['annot2'][0, 0]
			p3d = data['annot3'][0, 0]
			x2d.append(p2d)
			x3d.append(p3d)
			x2ds_sum += p2d.sum(-1)
			x2ds_sqr_sum += np.power(p2d, 2).sum(-1)
			x3ds_sum += p3d.sum(-1)
			x3ds_sqr_sum += np.power(p3d, 2).sum(-1)
			k += (17)
		x2ds.append(np.array(x2d, dtype=np.double))
		x3ds.append(np.array(x3d, dtype=np.double))
	x2ds_mean = x2ds_sum / k
	x2ds_std = np.sqrt(x2ds_sqr_sum / k - x2ds_mean ** 2)
	x3ds_mean = x3ds_sum / k
	x3ds_std = np.sqrt(x3ds_sqr_sum / k - x3ds_mean ** 2)
	## Mask File 
	mask_mat = sio.loadmat(file_path_2)['occlusion_labels']
	ms, ts = [], []
	for t in range(mask_mat.shape[0]):
		m = []
		for p in range(mask_mat.shape[1]):
			mask = 1 - mask_mat[t][p]
			m.append(mask)
		ms.append(np.array(m))
		ts.append(t / fps)
	#return np.array(x2ds, dtype=np.double), np.array(x3ds, dtype=np.double), np.array(ms, dtype=np.int32), np.array(ts, dtype=np.double)
	return (np.array(x2ds, dtype=np.double) - np.expand_dims(x2ds_mean, axis=(0, 1, 3))) / np.expand_dims(x2ds_std, axis=(0, 1, 3)), (np.array(x3ds, dtype=np.double) - np.expand_dims(x3ds_mean, axis=(0, 1, 3))) / np.expand_dims(x3ds_std, axis=(0, 1, 3)), np.array(ms, dtype=np.int32), np.array(ts, dtype=np.double)


def split_data(file_path="../Data/MuPoTS/raw/", save_path="../Data/MuPoTS/processed/", seg_len=50, fps=30, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
	with h5py.File(os.path.join(save_path, 'data_{}.hdf5').format(seg_len), 'w') as Dataset: 
		## Static Adj Mat
		connection = np.array([
						[1, 16], [16, 2], 
						[2, 3], [3, 4], [4, 5], 
						[2, 6], [6, 7], [7, 8], 
						[2, 15], 
						[15, 9], [9, 10], [10, 11], 
						[15, 12], [12, 13], [13, 14]
					 ])
		static_adj = np.zeros((16, 16))
		for e in connection:
			static_adj[e[0] - 1][e[1] - 1] = 1.0
		static_adj = static_adj + static_adj.T
		Dataset.create_dataset('static_adj', data=static_adj)
		## Hierarchical Group
		group = [
					np.array([15]), 
					np.array([15, 2, 3, 6, 9, 12]),
					np.array([[2, 16, 1], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]])
				]
		for g in range(3):
			Dataset.create_dataset('group_{}'.format(g + 1), data=group[g])
		## Hierarchical Static Adj Mat
		static_adj_lv2 = np.zeros((6, 6))
		for e in connection:
			if e[0] in group[1] and e[1] in group[1]:
				static_adj_lv2[np.where(group[1] == e[0])[0].item()][np.where(group[1] == e[1])[0].item()] = 1.0
		static_adj_lv2 = static_adj_lv2 + static_adj_lv2.T
		Dataset.create_dataset('static_adj_lv2', data=static_adj_lv2)
		static_adj_lv3 = np.zeros((5, 3, 3))
		for sg in range(5):
			for e in connection:
				if e[0] in group[2][sg] and e[1] in group[2][sg]:
					static_adj_lv3[sg][np.where(group[2][sg] == e[0])[0].item()][np.where(group[2][sg] == e[1])[0].item()] = 1.0
			static_adj_lv3[sg, :, :] = static_adj_lv3[sg, :, :] + static_adj_lv3[sg, :, :].T
		Dataset.create_dataset('static_adj_lv3', data=static_adj_lv3)
		data_idx = 0
		data_idx_lst = []
		x2ds_sum, x2ds_sqr_sum, x3ds_sum, x3ds_sqr_sum, k = 0., 0., 0., 0., 0
		## Read Mat File
		for fi in range(1, 21):
			data_file = os.path.join(file_path, 'annot_{}.mat').format(fi)
			mask_file = os.path.join(file_path, 'occlusion_{}.mat').format(fi)
			x2d, x3d, m, t = fetch_mat(data_file, mask_file, fps)
			start = 0
			while start + seg_len <= t.shape[0]:
				end = start + seg_len
				## 2D & 3D Coordinates
				coord_2d = np.zeros((seg_len, 3, 2, 16))
				coord_2d[:, :x2d.shape[1], :, :15] = x2d[start : end, :, :, :15]
				coord_2d[:, :x2d.shape[1], :, 15] = x2d[start : end, :, :, 16]
				coord_3d = np.zeros((seg_len, 3, 3, 16))
				coord_3d[:, :x3d.shape[1], :, :15] = x3d[start : end, :, :, :15]
				coord_3d[:, :x3d.shape[1], :, 15] = x3d[start : end, :, :, 16]
				Dataset.create_dataset('x2ds_{}'.format(data_idx), data=coord_2d.reshape((seg_len, -1, 16, 2)))
				Dataset.create_dataset('x3ds_{}'.format(data_idx), data=coord_3d.reshape((seg_len, -1, 16, 3)))
				## Mask
				mask = np.zeros((seg_len, 3, 16))
				mask[:, :x2d.shape[1], :15] = m[start : end, :, 0, :15]
				mask[:, :x2d.shape[1], 15] = m[start : end, :, 0, 16]
				Dataset.create_dataset('masks_{}'.format(data_idx), data=mask)
				## Flatten & Hierarchical Dynamic Adj Mat
				dynamic_adj = np.zeros((seg_len, x2d.shape[1], 16, 16))
				dynamic_adj_lv1 = np.zeros((seg_len, 3, 3))
				dynamic_adj_lv2 = np.zeros((seg_len, 3, 6, 6))
				dynamic_adj_lv3 = np.zeros((seg_len, 3, 5, 3, 3))
				for ti in range(seg_len):
					# Lv1
					for i in range(x2d.shape[1]):
						for j in range(x2d.shape[1]):
							if i != j:
								dynamic_adj_lv1[ti, i, j] = 1.0
					dynamic_adj_lv1[ti] = dynamic_adj_lv1[ti] + dynamic_adj_lv1[ti].T
					for pi in range(x2d.shape[1]):
						# Lv2
						dm = mask[ti, pi, group[1] - 1].reshape((1, -1))
						dynamic_adj_lv2[ti, pi, :] = static_adj_lv2 * np.matmul(dm.T, dm)
						# Lv3
						for sg in range(5):
							dm = mask[ti, pi, group[2][sg] - 1].reshape((1, -1))
							dynamic_adj_lv3[ti, pi, sg, :] = static_adj_lv3[sg] *  np.matmul(dm.T, dm)
						# Flatten
						dm = mask[ti, pi, :].reshape((1, -1))
						dynamic_adj[ti, pi, :] = static_adj * np.matmul(dm.T, dm)
						# Mean & Std
						#x2ds_sum += coord_2d
						#x2ds_sqr_sum += np.power(p2d, 2)
						#x3ds_sum += p3d
						#x3ds_sqr_sum += np.power(p3d, 2)
				Dataset.create_dataset('dynamic_adj_lv1_{}'.format(data_idx), data=dynamic_adj_lv1)
				Dataset.create_dataset('dynamic_adj_lv2_{}'.format(data_idx), data=dynamic_adj_lv2)
				Dataset.create_dataset('dynamic_adj_lv3_{}'.format(data_idx), data=dynamic_adj_lv3)
				Dataset.create_dataset('dynamic_adj_{}'.format(data_idx), data=dynamic_adj)
				## T
				Dataset.create_dataset('ts_{}'.format(data_idx), data=t[start : end])
				start += (seg_len // 2)
				data_idx_lst.append(np.array([data_idx, x2d.shape[1]]))
				data_idx += 1
			print(data_file, ' Done!')
		shuffle(data_idx_lst)
		train_multi_lst = data_idx_lst[:int(train_ratio * data_idx)]
		valid_multi_lst = data_idx_lst[int(train_ratio * data_idx):int((train_ratio + valid_ratio) * data_idx)]
		test_multi_lst = data_idx_lst[int((train_ratio + valid_ratio) * data_idx):]
		Dataset.create_dataset('train_multi_idx', data=np.array(train_multi_lst, dtype=np.int32))
		Dataset.create_dataset('valid_multi_idx', data=np.array(valid_multi_lst, dtype=np.int32))
		Dataset.create_dataset('test_multi_idx', data=np.array(test_multi_lst, dtype=np.int32))
		train_single_lst = []
		for idx in train_multi_lst:
			for pi in range(idx[1]):
				train_single_lst.append(np.array([idx[0], pi]))
		shuffle(train_single_lst)
		valid_single_lst = []
		for idx in valid_multi_lst:
			for pi in range(idx[1]):
				valid_single_lst.append(np.array([idx[0], pi]))
		shuffle(valid_single_lst)
		test_single_lst = []
		for idx in test_multi_lst:
			for pi in range(idx[1]):
				test_single_lst.append(np.array([idx[0], pi]))
		Dataset.create_dataset('train_single_idx', data=np.array(train_single_lst, dtype=np.int32))
		Dataset.create_dataset('valid_single_idx', data=np.array(valid_single_lst, dtype=np.int32))
		Dataset.create_dataset('test_single_idx', data=np.array(test_single_lst, dtype=np.int32))
	return
	

def correct_data(data_path="../Data/MuPoTS/processed/data_50.hdf5"):
	with h5py.File(data_path, 'w') as Dataset: 
		test_single_lst = []
		test_multi_lst = Dataset['test_single_idx']
		for idx in test_multi_lst:
			for pi in range(idx[1]):
				test_single_lst.append(np.array([idx[0], pi]))
		Dataset['test_single_idx'] = np.array(test_single_lst, dtype=np.int32)
	return


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
			x2d = self.data['x2ds_{}'.format(index[0])][:]
			x3d = self.data['x3ds_{}'.format(index[0])][:]
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


#fetch_mat("../../Data/MuPoTS/raw/annot_1.mat", "../../Data/MuPoTS/raw/occlusion_1.mat")
#split_data()
#correct_data()