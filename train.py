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
from trainer import Trainer
import torch.nn.functional as F
import sys, torch, os.path

model_name = sys.argv[1]
pkls_dir = sys.argv[2]
dev = sys.argv[3]
DEVICE = torch.device("cuda" if torch.cuda.is_available() and dev=='cuda' else "cpu")
print(model_name, pkls_dir, DEVICE)

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
	train_datasets= PairedData(datadir='data/Train/Toled', npy=True)
	eval_datasets= PairedData(datadir='data/Eval/Toled', npy=True)
	train_loader = DataLoader(train_datasets, batch_size=16, shuffle=True)
	eval_loader = DataLoader(eval_datasets, batch_size=1, shuffle=False)
	# create model for Image-Restoration
	trainer = Trainer(train_data_loader=train_loader,
	                  eval_data_loader=eval_loader,
	                  network=model,
	                  loss_function=F.l1_loss,
	                  epoch=50,
	                  pkls_dir=os.path.join(pkls_dir, model_name),
	                  summary_dir='./summarylogs/{}'.format(model_name))
	model = trainer.train()