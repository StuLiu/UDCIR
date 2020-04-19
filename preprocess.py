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
from tqdm import tqdm

def read_imgs_from_dir(img_dir_path='data/Train/Toled/HQ')->np.ndarray:
	print('>>> read images from {}. <<<'.format(img_dir_path))
	if not os.path.exists(img_dir_path):
		raise Exception(" No such file or directory:{0}".format(img_dir_path))
	data_dir = os.path.join(os.getcwd(), img_dir_path)
	img_paths = glob(os.path.join(data_dir, '*.png'))
	img_paths.sort()
	img_data_list = []
	for path in tqdm(img_paths):
		img_data = cv2.imread(path)    # return the ndarray of image, [high, width, channal(3)]
		img_data_list.extend(_image_enhance(img_data))
	return np.array(img_data_list)

def _image_crop(img)->list:
	assert img.shape[0]==1024 and img.shape[1]==2048, 'The img is not a UDC photo'
	Width = 256
	results = []
	for i in range(4):
		for j in range(8):
			results.append(img[i*Width:(i+1)*Width, j*Width:(j+1)*Width, :])
	print(results[0].shape)
	return results

def _image_rotate(img)->list:
	"""ratate the img with 90°, 180°, and 270°, separately"""
	img_cw_90  = cv2.flip(cv2.transpose(img), 1)
	img_cw_180 = cv2.flip(img, -1)
	img_cw_270 = cv2.flip(cv2.transpose(img), 0)
	return [img, img_cw_90, img_cw_180, img_cw_270]

def _image_enhance(img)->list:
	imgs_cropped = _image_crop(img)
	results = []
	for img_ in imgs_cropped:
		img_flipped = cv2.flip(img_, 1)     # flip the image around y-axis
		results.extend(_image_rotate(img_))
		results.extend(_image_rotate(img_flipped))
	return results

if __name__ == '__main__':
	HQ = read_imgs_from_dir(img_dir_path='data/Train/Toled/HQ')
	LQ = read_imgs_from_dir(img_dir_path='data/Train/Toled/LQ')
	np.save('data/Train/Toled/HQ/HQ.npy', HQ)
	np.save('data/Train/Toled/HQ/LQ.npy', LQ)
	print(HQ.shape, LQ.shape)
	# cv2.imshow('HQ', HQ[22])
	# cv2.waitKey(0)
	# cv2.imshow('LQ', LQ[22])
	# cv2.waitKey(0)


