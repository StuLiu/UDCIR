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
from model import UNet, CNN
from tester import Tester
from utils import compute_PSNR
import torch, sys

model_name = sys.argv[1]
pkl_dir = sys.argv[2]
dev = sys.argv[3]
DEVICE = torch.device("cuda" if torch.cuda.is_available() and dev=='cuda' else "cpu")
print(model_name, pkl_dir, DEVICE)

if model_name == 'UNet-16':
	model = UNet(N=16)
elif model_name == 'UNet-32':
	model = UNet(N=32)
elif model_name == 'UNet-64':
	model = UNet(N=64)
elif model_name == 'CNN':
	model = CNN(N=32)
else:
	model = None
	print('>>> Model name:{} invalid!')
	exit(-1)
# run "python train.py ./pkls UNet-16"

if __name__ == '__main__':
	# load train data
	test_datasets= PairedData(datadir='data/Eval/Toled', npy=False)
	print(len(test_datasets))
	test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False)
	# create model for Image-Restoration
	model.load_state_dict(torch.load(
		f=pkl_dir,
		map_location=DEVICE))
	tester = Tester(dataloader=test_loader,
	                network=model,
	                functions=[compute_PSNR],
	                device=DEVICE)
	tester.test()