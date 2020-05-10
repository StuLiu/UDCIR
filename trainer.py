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
import sys, os
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils import compute_PSNR

class Trainer(object):
	""" The class to train networks"""
	def __init__(self,
	             train_data_loader,
	             eval_data_loader,
	             network,
	             optimizer=torch.optim.Adam,
	             learning_rate=1.0e-4,
	             epoch=1000,
	             loss_function=F.mse_loss,
	             pkls_path='./pkls/'):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print('device:', self.device)
		self.train_data_loader = train_data_loader
		self.eval_data_loader = eval_data_loader
		print(len(self.train_data_loader), len(self.eval_data_loader))
		self.net = network.to(self.device)
		self.lr = learning_rate
		self.opt = optimizer(self.net.parameters(), lr=self.lr)
		self.epoch = epoch
		self.loss_F = loss_function
		self.pkls_path = pkls_path
		if not os.path.exists(self.pkls_path):
			os.makedirs(self.pkls_path)
		self.scheduler = lr_scheduler.StepLR(self.opt, step_size=2, gamma=0.9)

	def train(self):
		print('Do training...')
		self.net.train()
		for epoch in range(1, self.epoch + 1):
			for batch_idx, (data, target) in enumerate(self.train_data_loader):
				data, target = data.to(self.device), target.to(self.device)
				self.opt.zero_grad()
				output = self.net(data)
				loss_batch = self.loss_F(output, target)
				loss_batch.backward()
				self.opt.step()
				if (batch_idx + 1) % 20 == 0:
					self._eval_and_save(epoch, batch_idx + 1)
			sys.stdout.write('\n')
			self.scheduler.step(epoch)
		return self.net

	def _eval_and_save(self, epoch, batch_idx):
		batch_idx_global = batch_idx + (epoch - 1) * len(self.train_data_loader)
		with torch.no_grad():
			eval_loss_sum = 0
			for batch_idx_eval, (data_eval, target_eval) in enumerate(self.eval_data_loader):
				data_eval, target_eval = data_eval.to(self.device), target_eval.to(self.device)
				output_eval = self.net(data_eval)
				eval_loss_sum += self.loss_F(output_eval, target_eval).item()
			eval_loss = eval_loss_sum / len(self.eval_data_loader)
			sys.stdout.write('\rTrain Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.8f}'.format(
				epoch, batch_idx, len(self.train_data_loader),
				100. * batch_idx / len(self.train_data_loader),
				eval_loss))
			with SummaryWriter(log_dir='./summarylogs', comment='train') as writer:
				writer.add_scalar('lr', self.opt.state_dict()['param_groups'][0]['lr'], batch_idx_global)
				writer.add_scalar('Loss', eval_loss, batch_idx_global)
				writer.add_scalar(
					'PSNR', compute_PSNR(target_eval.cpu().numpy(), output_eval.cpu().numpy()),
					batch_idx_global
				)
			torch.save(self.net.state_dict(), os.path.join(self.pkls_path,
			                                               'model_{}.pkl'.format(batch_idx_global)))

