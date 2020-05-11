'''
--------------------------------------------------------
@File    :   train.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2020/4/20 21:14     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
''' 

from torch.utils.data import DataLoader
from dataloader import PairedData
from model import Generator, UNet
from tester import Tester
from utils import compute_PSNR
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
	# load train data
	test_datasets= PairedData(datadir='data/Train/Toled', npy=False)
	print(len(test_datasets))
	test_loader = DataLoader(test_datasets, batch_size=16, shuffle=False)
	# create model for Image-Restoration
	model = UNet().to(DEVICE)
	model.load_state_dict(torch.load(
		f='pkls/UNet/model_8960.pkl',
		map_location=DEVICE))
	tester = Tester(dataloader=test_loader, network=model, functions=[compute_PSNR])
	tester.test()