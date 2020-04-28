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
from model import Restorer
from trainer import Trainer
import torch.nn.functional as F

if __name__ == '__main__':
	# load train data
	train_datasets= PairedData(datadir='data/Train/Toled')
	eval_datasets= PairedData(datadir='data/Eval/Toled')
	train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
	eval_loader = DataLoader(eval_datasets, batch_size=64, shuffle=False)
	# create model for Image-Restoration
	model = Restorer(image_c=3, N=64)
	trainer = Trainer(train_data_loader=train_loader,
	                  eval_data_loader=eval_loader,
	                  network=model,
	                  loss_function=F.l1_loss,
	                  epoch=40)
	model = trainer.train()