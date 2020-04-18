'''
--------------------------------------------------------
@File    :   utils.py    
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

def read_imgs_from_dir(img_dir_path='data/trai/Toled/HQ')->np.array:
	print('>>> read images from {}. <<<'.format(img_dir_path))
	if not os.path.exists(img_dir_path):
		raise Exception(" No such file or directory:{0}".format(img_dir_path))
	data_dir = os.path.join(os.getcwd(), img_dir_path)
	img_paths = glob(os.path.join(data_dir, '*.png'))
	img_paths.sort()
	img_data_list = []
	for p in tqdm(img_paths):
		img_data = cv2.imread(p)
		img_data_list.append(img_data)
	return np.array(img_data_list)

if __name__ == '__main__':
	print(read_imgs_from_dir(img_dir_path='data/train/Toled/HQ').shape)