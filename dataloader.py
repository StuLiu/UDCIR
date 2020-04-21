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
from torch.utils.data import DataLoader

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

if __name__ == '__main__':
	# load train data
	myDatasets= PairedData(datadir='data/Train/Toled')
	train_loader = DataLoader(myDatasets, batch_size=32, shuffle=True)
	x_eval, y_eval = [], []
	for i, (x, y) in enumerate(train_loader):
		x_eval.append(x)
		y_eval.append(y)
		if i >= 4:
			break
	x_eval = np.array(x_eval).reshape((-1, 3, 128, 128))
	y_eval = np.array(y_eval).reshape((-1, 3, 128, 128))
	print(x_eval.shape, y_eval.shape)
