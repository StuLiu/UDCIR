'''
--------------------------------------------------------
@File    :   dataloader.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2020/4/17 12:37     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
''' 
from torch.utils.data import Dataset
import os.path
import numpy as np
from torch import from_numpy

class PairedData(Dataset):
	def __init__(self, datadir='data/Train/Toled'):
		self.X = np.load(os.path.join(datadir, 'LQ.npy'))
		self.Y = np.load(os.path.join(datadir, 'HQ.npy'))
		assert self.X.shape == self.Y.shape, 'data unpaired'
		self.datasize = len(self.X)
		print('Loaded {} paired data from {}.'.format(self.datasize, datadir))

	def __len__(self):
		return self.datasize

	def __getitem__(self, idx):
		return from_numpy(self.X[idx]).float(), from_numpy(self.Y[idx]).float()