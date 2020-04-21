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

def compute_PSNR(img1, img2)->float:
	assert img1.shape == img2.shape
	# img1 and img2 have range [0, 255], numpy type
	mse = np.mean((img1 - img2) ** 2)
	if mse == 0:
		return float('inf')
	return 20 * math.log10(255.0 / math.sqrt(mse))