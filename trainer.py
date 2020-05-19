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
import sys, os, time
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import compute_PSNR, keep_newest

class Trainer(object):
	""" The class to train networks"""
	def __init__(self,
	             train_dataset,
	             eval_dataset,
	             network,
	             optimizer=torch.optim.Adam,
	             batch_size=16,
	             learning_rate=1.0e-4,
	             epoch=400,
	             loss_function=F.mse_loss,
	             pkls_dir='./pkls/',
	             summary_dir='./summarylogs/',):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print('device:', self.device)
		self.train_dataset = train_dataset
		self.train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
		self.eval_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
		print(len(self.train_data_loader), len(self.eval_data_loader))
		self.net = network.to(self.device)
		self.batch_size = batch_size
		self.lr = learning_rate
		self.opt = optimizer(self.net.parameters(), lr=self.lr)
		self.epoch = epoch
		self.loss_F = loss_function
		self.pkls_dir = pkls_dir
		self.summary_dir = summary_dir
		if not os.path.exists(self.pkls_dir):
			os.makedirs(self.pkls_dir)
		if not os.path.exists(self.summary_dir):
			os.makedirs(self.summary_dir)
		self.scheduler = lr_scheduler.StepLR(self.opt, step_size=100, gamma=0.1)
		print(self.batch_size, self.lr, self.epoch)

	def train(self):
		print('Do training...')
		sys.stdout.flush()
		self.net.train()
		for epoch in range(1, self.epoch + 1):
			for batch_idx, (data, target) in enumerate(self.train_data_loader):
				data, target = data.to(self.device), target.to(self.device)
				self.opt.zero_grad()
				output = self.net(data)
				loss_batch = self.loss_F(output, target)
				loss_batch.backward()
				self.opt.step()
				batch_idx_global = batch_idx + (epoch - 1) * len(self.train_data_loader)
				if (batch_idx_global + 1) % 40 == 0 or batch_idx_global == 0:
					self._eval_and_save(epoch, batch_idx)
			sys.stdout.write('\n')
			self.scheduler.step(epoch)
			self.train_dataset.shuffle()
			self.train_data_loader = DataLoader(self.train_dataset,
			                                    batch_size=self.batch_size,
			                                    shuffle=False)
		return self.net

	def _eval_and_save(self, epoch, batch_idx):
		batch_idx_global = batch_idx + (epoch - 1) * len(self.train_data_loader)
		with torch.no_grad():
			eval_loss_sum, eval_psnr_sum = 0, 0
			for batch_idx_eval, (data_eval, target_eval) in enumerate(self.eval_data_loader):
				data_eval, target_eval = data_eval.to(self.device), target_eval.to(self.device)
				output_eval = self.net(data_eval)
				eval_loss_sum += self.loss_F(output_eval * 255, target_eval * 255).item()
				eval_psnr_sum += compute_PSNR(output_eval.cpu().numpy() * 255,
				                              target_eval.cpu().numpy() * 255)
			eval_loss = eval_loss_sum / len(self.eval_data_loader)
			eval_psnr = eval_psnr_sum / len(self.eval_data_loader)
			sys.stdout.write('\r{} Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.4f}\tPSNR:{:.2f}'
				.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
			            epoch, batch_idx + 1, len(self.train_data_loader),
			            100. * (batch_idx + 1) / len(self.train_data_loader),
			            eval_loss, eval_psnr))
			with SummaryWriter(log_dir=self.summary_dir, comment='train') as writer:
				writer.add_scalar('lr', self.opt.state_dict()['param_groups'][0]['lr'], batch_idx_global + 1)
				writer.add_scalar('Loss', eval_loss, batch_idx_global + 1)
				writer.add_scalar('PSNR', eval_psnr, batch_idx_global + 1)
			torch.save(self.net.state_dict(), os.path.join(self.pkls_dir,
			                                               'model_{}.pkl'.format(batch_idx_global + 1)))
			keep_newest(dir_path=self.pkls_dir, k=150)
			sys.stdout.flush()