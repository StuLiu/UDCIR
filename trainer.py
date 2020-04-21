'''
--------------------------------------------------------
@File    :   trainer.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2020/4/16 13:35     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
'''
import sys
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import compute_PSNR

class Trainer(object):
	""" The class to train networks"""
	def __init__(self,
	             dataloader,
	             network,
	             optimizer=torch.optim.Adam,
	             learning_rate=1.0e-4,
	             epoch=1000,
	             loss_function=F.mse_loss):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print('device:', self.device)
		self.dataloader = dataloader
		self.net = network.to(self.device)
		self.lr = learning_rate
		self.opt = optimizer(self.net.parameters(), lr=self.lr)
		self.epoch = epoch
		self.loss_F = loss_function

	def train(self):
		self.net.train()
		for epoch in range(1, self.epoch + 1):
			for batch_idx, (data, target) in enumerate(self.dataloader):
				data, target = data.to(self.device), target.to(self.device)
				self.opt.zero_grad()
				output = self.net(data)
				loss_batch = self.loss_F(output, target)
				loss_batch.backward()
				self.opt.step()
				if (batch_idx + 1) % 2 == 0:
					sys.stdout.write('\rTrain Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
						epoch, batch_idx * len(data), len(self.dataloader.dataset),
						100. * batch_idx / len(self.dataloader), loss_batch.item()))
					with SummaryWriter(log_dir='./summarylogs', comment='train') as writer:
						batch_idx_global = batch_idx + (epoch - 1) * len(self.dataloader)
						writer.add_scalar('Loss', loss_batch.item(), batch_idx_global)
						with torch.no_grad():
							writer.add_scalar(
								'PSNR', compute_PSNR(target.cpu().numpy(), output.cpu().numpy()),
						        batch_idx_global
							)
			sys.stdout.write('\n')
		return self.net