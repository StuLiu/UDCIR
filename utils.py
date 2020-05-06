'''
--------------------------------------------------------
@File    :   utils.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2020/4/21 11:46     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
'''
import math
import numpy as np
import os.path
import shutil
import torch
from scipy.io.matlab.mio import savemat, loadmat
from torch import from_numpy
import cv2


def compute_PSNR(img1, img2)->float:
	assert img1.shape == img2.shape
	# img1 and img2 have range [0, 255], numpy type
	mse = np.mean((img1 - img2) ** 2)
	if mse == 0:
		return float('inf')
	return 20 * math.log10(255.0 / math.sqrt(mse))

def show_mat(mat_file_path='./res_dir/results.mat', key='results'):
	mat_data = loadmat(mat_file_path)[key]
	print(mat_data.shape)
	for i in range(len(mat_data)):
		cv2.imshow('result', mat_data[i])
		cv2.waitKey(0)

if __name__ == '__main__':
	# show_mat(mat_file_path='./toled_val_display.mat', key='val_display')
	show_mat(mat_file_path='./res_dir/results.mat', key='results')