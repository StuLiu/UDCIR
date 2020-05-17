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
import random, cv2, time
from torch import from_numpy
from torch.utils.data import DataLoader
from data_io import read_imgs_from_dir

class EvalDataset(Dataset):
	def __init__(self, datadir='data/Train/Toled', npy=True):
		if npy:
			self.X = np.load(os.path.join(datadir, 'LQ.npy'))
			self.Y = np.load(os.path.join(datadir, 'HQ.npy'))
		else:
			x = read_imgs_from_dir(os.path.join(datadir, 'LQ'), enhance=False)
			y = read_imgs_from_dir(os.path.join(datadir, 'HQ'), enhance=False)
			self.X = np.transpose(x, (0, 3, 1, 2))
			self.Y = np.transpose(y, (0, 3, 1, 2))
		assert self.X.shape == self.Y.shape, 'data unpaired'
		self.datasize = len(self.X)
		print('Loaded {} EvalData from {}.'.format(self.datasize, datadir))
		print('shape of X and Y:{}.'.format(self.X.shape))

	def __len__(self):
		return self.datasize

	def __getitem__(self, idx):
		return from_numpy(self.X[idx]).float(), from_numpy(self.Y[idx]).float()

class TrainDataset(Dataset):
	def __init__(self, width=256, datadir='data/Train/Toled', npy=True):
		self.width = width
		if npy:
			# read images from .npy, (N, c, h, w), (N, c, h, w)
			x_origin = np.load(os.path.join(datadir, 'LQ.npy'))
			y_origin = np.load(os.path.join(datadir, 'HQ.npy'))
			self.x_origin = np.transpose(x_origin, (0, 2, 3, 1))
			self.y_origin = np.transpose(y_origin, (0, 2, 3, 1))
		else:
			# read images from file system, (N, h, w, c), (N, h, w, c)
			self.x_origin = read_imgs_from_dir(os.path.join(datadir, 'LQ'), enhance=False)
			self.y_origin = read_imgs_from_dir(os.path.join(datadir, 'HQ'), enhance=False)
		self.X, self.Y = self._shuffle(self.x_origin, self.y_origin)
		assert self.X.shape == self.Y.shape, 'data unpaired'
		self.datasize = len(self.X)
		print('Loaded {} TrainData from {}.'.format(self.datasize, datadir))
		print('shape of X and Y:{}.'.format(self.X.shape))

	def __len__(self):
		return self.datasize

	def __getitem__(self, idx):
		return from_numpy(self.X[idx]).float(), from_numpy(self.Y[idx]).float()

	def shuffle(self):
		self.X, self.Y = self._shuffle(self.x_origin, self.y_origin)

	def _shuffle(self, x:np.ndarray, y:np.ndarray)->(np.ndarray, np.ndarray):
		""" image enhance: random crop, flip, and rotate
		:param x: LQ images ndarray with shape (N, h, w, c)
		:param y: HQ images ndarray with shape (N, h, w, c)
		:return: LQ and HQ image blocks ndarrays width shape (N, c, self.width, self.width)
		"""
		assert x.shape == y.shape, 'shape of x and y disaffinity'
		X, Y = [], []
		# for each image pairs
		for i in range(len(x)):
			index_h = random.randint(0, x.shape[1] - self.width)
			index_w = random.randint(0, x.shape[2] - self.width)
			x_block = x[i, index_h: index_h + self.width, index_w: index_w + self.width, :]
			y_block = y[i, index_h: index_h + self.width, index_w: index_w + self.width, :]
			# print(x_block.shape, y_block.shape)
			X.extend(self._image_enhance(x_block))
			Y.extend(self._image_enhance(y_block))
		X = np.transpose(np.array(X), (0, 3, 1, 2))
		Y = np.transpose(np.array(Y), (0, 3, 1, 2))
		seed = int(time.time())
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(Y)
		return X, Y

	def _image_enhance(self, img: np.ndarray)->list:
		results = []
		# flip around y axis
		img_flip_y = cv2.flip(img, 1)
		results.extend(self._image_rotate(img))
		results.extend(self._image_rotate(img_flip_y))
		return results

	def _image_rotate(self, img: np.ndarray) -> list:
		"""ratate the img with 90°, 180°, and 270°, separately"""
		img_cw_90 = cv2.flip(cv2.transpose(img), 1)
		img_cw_180 = cv2.flip(img, -1)
		img_cw_270 = cv2.flip(cv2.transpose(img), 0)
		return [img, img_cw_90, img_cw_180, img_cw_270]

# np.ndarray.astype()
if __name__ == '__main__':
	# load train data
	myDatasets= TrainDataset(datadir='data/Eval/Toled', npy=False)
	while True:
		train_loader = DataLoader(myDatasets, batch_size=32, shuffle=False)
		x_eval, y_eval = [], []
		for i, (x, y) in enumerate(train_loader):
			lq, hq = np.transpose(x.cpu().numpy(), (0, 2, 3, 1)), np.transpose(y.cpu().numpy(), (0, 2, 3, 1))
			print(lq.shape, hq.shape)
			print(lq[0], hq[0])
			print(type(lq), type(hq))
			for j in range(len(x)):
				print(lq[j].shape, hq[j].shape)
				print(lq[j], hq[j])
				print(type(lq[j]), type(hq[j]))
				cv2.imshow('LQ', np.array(lq[j].astype('uint8')))
				cv2.waitKey(0)
				cv2.imshow('HQ', np.array(hq[j].astype('uint8')))
				cv2.waitKey(0)
				break
			break
		myDatasets.shuffle()
	# 	x_eval.append(x.numpy())
	# 	y_eval.append(y.numpy())
	# 	if i >= 4:
	# 		break
	# x_eval = np.array(x_eval).reshape((-1, 3, 256, 256))
	# y_eval = np.array(y_eval).reshape((-1, 3, 256, 256))
	# print(x_eval.shape, y_eval.shape)
	# np.save('data/Eval/Toled/LQ_256.npy', x_eval)
	# np.save('data/Eval/Toled/HQ_256.npy', y_eval)
