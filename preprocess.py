'''
--------------------------------------------------------
@File    :   preprocess.py
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2020/4/18 10:57     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
''' 
import os
from glob import glob
import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

def read_imgs_from_dir(img_dir_path='data/Train/Toled/HQ', enhance=True)->np.ndarray:
	print('>>> read images from {}. <<<'.format(img_dir_path))
	if not os.path.exists(img_dir_path):
		raise Exception(" No such file or directory:{0}".format(img_dir_path))
	data_dir = os.path.join(os.getcwd(), img_dir_path)
	img_paths = glob(os.path.join(data_dir, '*.png'))
	img_paths.sort()
	img_data_list = []
	for path in tqdm(img_paths):
		img_data = cv2.imread(path)    # return the ndarray of image, [high, width, channal(3)]
		if enhance:
			img_data_list.extend(_image_enhance(img_data))
		else:
			img_data_list.extend([img_data])
	return np.array(img_data_list)

def read_imgs_from_mat(mat_file_path='data/Train/Toled/HQ')->np.ndarray:
	data_dict = loadmat(mat_file_path)
	imgs_data = data_dict['val_display']
	return np.array(imgs_data)

def _image_crop(img)->list:
	assert img.shape[0]==1024 and img.shape[1]==2048, 'The img is not a UDC photo'
	Width = 128
	results = []
	for i in range(8):
		for j in range(16):
			results.append(img[i*Width:(i+1)*Width, j*Width:(j+1)*Width, :])
	return results

def _image_rotate(img)->list:
	"""ratate the img with 90°, 180°, and 270°, separately"""
	img_cw_90  = cv2.flip(cv2.transpose(img), 1)
	img_cw_180 = cv2.flip(img, -1)
	img_cw_270 = cv2.flip(cv2.transpose(img), 0)
	return [img, img_cw_90, img_cw_180, img_cw_270]

def _image_enhance(img)->list:
	imgs_cropped = _image_crop(img)
	# return imgs_cropped
	results = []
	for img_ in imgs_cropped:
		# img_flipped_y = cv2.flip(img_, 1)       # flip the image around y-axis
		# img_flipped_x = cv2.flip(img_, 0)       # flip the image around x-axis
		img_flipped_xy = cv2.flip(img_, -1)  # flip the image around x-axis
		results.extend([img_, img_flipped_xy])
		# results.extend([img_, img_flipped_y, img_flipped_x, img_flipped_xy])
	print(np.array(results).shape)
	return results

def _imgs2npy(img_dir_path, enhance, out_path):
	imgs = read_imgs_from_dir(img_dir_path, enhance=enhance)  # (N, h, w, c)
	imgs = np.transpose(imgs, (0, 3, 1, 2))
	print(imgs.shape)
	np.save(out_path, imgs)
	# cv2.imshow('HQ', np.transpose(HQ[22], (1, 2, 0)))
	# cv2.waitKey(0)
	# cv2.imshow('LQ', np.transpose(LQ[22], (1, 2, 0)))
	# cv2.waitKey(0)
	print('saved to npy.')


if __name__ == '__main__':
	# print(read_imgs_from_mat('toled_val_display.mat').shape)
	# cv2.imshow('LQ', read_imgs_from_mat('toled_val_display.mat')[22])
	# cv2.waitKey(0)
	_imgs2npy(img_dir_path='data/Train/Toled/HQ', enhance=True, out_path='data/Train/Toled/HQ.npy')
	_imgs2npy(img_dir_path='data/Train/Toled/LQ', enhance=True, out_path='data/Train/Toled/LQ.npy')





