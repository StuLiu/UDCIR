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

def read_imgs_from_dir(img_dir_path='data/Train/Toled/HQ', enhance=True, only_crop=False)->np.ndarray:
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
			img_data_list.extend(image_enhance(img_data, only_crop=only_crop))
		else:
			img_data_list.extend([img_data])
	return np.array(img_data_list, dtype=np.uint8)

def read_imgs_from_mat(mat_file_path='data/Train/Toled/HQ')->np.ndarray:
	data_dict = loadmat(mat_file_path)
	imgs_data = data_dict['val_display']
	return np.array(imgs_data)

def image_crop(img)->list:
	""" crop image with size 1024*2048
	:param img:  img.shape: (h, w, c)
	:return:  results list: (N, h, w, c)
	"""
	assert img.shape[0]==1024 and img.shape[1]==2048, 'The img is not a UDC photo'
	Width = 256
	results = []
	for i in range(4):
		for j in range(8):
			results.append(img[i*Width:(i+1)*Width, j*Width:(j+1)*Width, :])
	# cv2.imshow('w', results[0])
	# cv2.waitKey(0)
	return results          # (N, h, w, c)

def image_splice(imgs:np.ndarray)->np.ndarray:
	""" splice images into a complete image with size of 1024*2048
	:param img: img.shape: (h, w, c)
	:return:  results list: (N, h, w, c)
	"""
	Width = 256
	assert imgs.shape==(32, Width, Width, 3), 'The imgs are invalid.'
	spliced_row_list = []
	for i in range(4):
		spliced_row_list.append(np.concatenate([img for img in imgs[i * 8: (i + 1) * 8]], axis=1))
	return np.concatenate(spliced_row_list, axis=0)          # (h, w, c)

def _image_rotate(img)->list:
	"""ratate the img with 90°, 180°, and 270°, separately"""
	img_cw_90  = cv2.flip(cv2.transpose(img), 1)
	img_cw_180 = cv2.flip(img, -1)
	img_cw_270 = cv2.flip(cv2.transpose(img), 0)
	return [img, img_cw_90, img_cw_180, img_cw_270]

def image_enhance(img, only_crop=False)->list:
	imgs_cropped = image_crop(img)
	if only_crop:
		return imgs_cropped
	else:
		results = []
		for img_ in imgs_cropped:
			img_flipped_y = cv2.flip(img_, 1)         # flip the image around y-axis
			img_flipped_x = cv2.flip(img_, 0)         # flip the image around x-axis
			img_flipped_xy = cv2.flip(img_, -1)       # flip the image around x- and y-axis
			# img_transposed = cv2.transpose(img)     # transpose the image
			results.extend([img_, img_flipped_xy, img_flipped_x, img_flipped_y])
		# print(np.array(results).shape)
		return results

def _imgs2npy(img_dir_path, out_path, enhance, only_crop):
	imgs = read_imgs_from_dir(img_dir_path,
	                          enhance=enhance,
	                          only_crop=only_crop)      # (N, h, w, c)
	print(imgs.shape)
	np.save(out_path, imgs)
	# cv2.imshow('HQ', HQ[22])
	# cv2.waitKey(0)
	# cv2.imshow('LQ', LQ[22])
	# cv2.waitKey(0)
	print('saved to npy.')


if __name__ == '__main__':
	# print(read_imgs_from_mat('toled_val_display.mat').shape)
	# cv2.imshow('LQ', read_imgs_from_mat('toled_val_display.mat')[22])
	# cv2.waitKey(0)
	# _imgs2npy(img_dir_path='data/Train/Toled/HQ',
	#           out_path='data/Train/Toled/HQ.npy',
	#           enhance=False,
	#           only_crop=False)
	# _imgs2npy(img_dir_path='data/Train/Toled/LQ',
	#           out_path='data/Train/Toled/LQ.npy',
	#           enhance=False,
	#           only_crop=False)
	# _imgs2npy(img_dir_path='data/Eval/Toled/HQ',
	#           out_path='data/Eval/Toled/HQ.npy',
	#           enhance=False,
	#           only_crop=False)
	# _imgs2npy(img_dir_path='data/Eval/Toled/LQ',
	#           out_path='data/Eval/Toled/LQ.npy',
	#           enhance=False,
	#           only_crop=False)
	_imgs2npy(img_dir_path='data/Train/Poled/HQ',
			  out_path='data/Train/Poled/HQ.npy',
			  enhance=False,
			  only_crop=False)
	_imgs2npy(img_dir_path='data/Train/Poled/LQ',
			  out_path='data/Train/Poled/LQ.npy',
			  enhance=False,
			  only_crop=False)
	_imgs2npy(img_dir_path='data/Eval/Poled/HQ',
			  out_path='data/Eval/Poled/HQ.npy',
			  enhance=False,
			  only_crop=False)
	_imgs2npy(img_dir_path='data/Eval/Poled/LQ',
			  out_path='data/Eval/Poled/LQ.npy',
			  enhance=False,
			  only_crop=False)




