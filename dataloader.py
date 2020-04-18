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
from utils import read_imgs_from_dir
import os.path as pt

class PairedData(Dataset):
	def __init__(self, datadir='data/train/Toled'):
		self.X = read_imgs_from_dir(img_dir_path=pt.join(datadir, 'LQ'))
		self.Y = read_imgs_from_dir(img_dir_path=pt.join(datadir, 'HQ'))
		assert self.X.shape == self.Y.shape, 'data unpaired'
		self.datasize = len(self.X)

	def __len__(self):
		return self.datasize

	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx]