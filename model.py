'''
--------------------------------------------------------
@File    :   model.py
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2020/4/16 13:34     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
''' 
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
	""" The UNet module"""
	def __init__(self, input_c=3, output_c=3, N=64):
		super(UNet, self).__init__()
		# def the operations in UNet
		self.step_0 = self._step(in_c=input_c, out_c=N, N=N)
		self.down_0 = nn.Conv2d(N, N*2, kernel_size=3, stride=2, padding=1)
		self.step_1 = self._step(in_c=N*2, out_c=N*2, N=N*2)
		self.down_1 = nn.Conv2d(N*2, N*4, kernel_size=3, stride=2, padding=1)
		self.step_2 = self._step(in_c=N*4, out_c=N*4, N=N*4)
		self.down_2 = nn.Conv2d(N*4, N*8, kernel_size=3, stride=2, padding=1)
		self.step_3 = self._step(in_c=N*8, out_c=N*8, N=N*8)
		self.step_4 = self._step(in_c=N*8, out_c=N*8, N=N*8)
		self.up_0 = nn.ConvTranspose2d(N*16, N*4, kernel_size=4, stride=2, padding=1)
		self.step_5 = self._step(in_c=N*4, out_c=N*4, N=N*4)
		self.up_1 = nn.ConvTranspose2d(N*8, N*2, kernel_size=4, stride=2, padding=1)
		self.step_6 = self._step(in_c=N*2, out_c=N*2, N=N*2)
		self.up_2 = nn.ConvTranspose2d(N*4, N, kernel_size=4, stride=2, padding=1)
		self.step_7 = self._step(in_c=N, out_c=output_c, N=N)

	def forward(self, x):
		x_step_0 = self._forward_step(x, self.step_0)
		x_down_0 = F.leaky_relu(self.down_0(x_step_0), 0.2)
		x_step_1 = self._forward_step(x_down_0, self.step_1)
		x_down_1 = F.leaky_relu(self.down_0(x_step_1), 0.2)
		x_step_2 = self._forward_step(x_down_1, self.step_2)
		x_down_2 = F.leaky_relu(self.down_0(x_step_2), 0.2)
		x_step_3 = self._forward_step(x_down_2, self.step_3)
		x_step_4 = self._forward_step(x_step_3, self.step_4)
		x_up_0 = F.leaky_relu(self.up_0(torch.cat([x_down_2, x_step_4], dim=1)), 0.2)
		x_step_5 = self._forward_step(x_up_0, self.step_5)
		x_up_1 = F.leaky_relu(self.up_1(torch.cat([x_down_1, x_step_5], dim=1)), 0.2)
		x_step_6 = self._forward_step(x_up_1, self.step_6)
		x_up_2 = F.leaky_relu(self.up_2(torch.cat([x_down_0, x_step_6], dim=1)), 0.2)
		x_step_7 = self._forward_step(x_up_2, self.step_7)
		return x + x_step_7

	def _step(self, in_c, out_c, N)->list:
		"""
		:param in_c:
		:param out_c:
		:param N:
		:return:
		"""
		step_operations = []
		step_operations.append(nn.Conv2d(in_c, N, kernel_size=3, stride=1, padding=1))
		step_operations.append(nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1))
		step_operations.append(nn.Conv2d(N, out_c, kernel_size=3, stride=1, padding=1))
		return step_operations

	def _forward_step(self, x, step:list)->torch.Tensor:
		"""
		:param x:
		:param step:
		:return:
		"""
		assert len(step) == 3, 'invalid step'
		x1 = F.leaky_relu(step[0](x), 0.2)
		x2 = F.leaky_relu(step[1](x1), 0.2) + x
		x3 = F.leaky_relu(step[2](x2), 0.2) + x1
		return x3


class Generator(nn.Module):
	""" The nueral network for real-time image restoration. """
	def __init__(self, image_c=3, N=64):
		super(Generator, self).__init__()
		# def the operations in network
		self.conv_0 = nn.Conv2d(image_c, N, (3, 3), stride=1, padding=1)
		self.relu_0 = nn.LeakyReLU(0.2)
		feature_extract_layers = OrderedDict()
		for i in range(5):
			feature_extract_layers['conv_{}'.format(i + 1)] = nn.Conv2d(
				N, N, (3, 3), stride=1, padding=1 )
			feature_extract_layers['relu_{}'.format(i + 1)] = nn.LeakyReLU(0.2)
		self.feature_extract_block = nn.Sequential(feature_extract_layers)
		self.conv_end = nn.Conv2d(N, image_c, (3, 3), stride=1, padding=1)

	def forward(self, x):
		fm_0 = self.relu_0(self.conv_0(x))
		fm_1 = self.feature_extract_block(fm_0)
		fm_end = self.conv_end(fm_1)
		return x + fm_end


class Discriminator(object):
	""" The nueral network for discriminating the real high-quality images and generated images. """
	def __init__(self, image_c=3, N=64):
		super(Discriminator, self).__init__()
		# def the operations in network
		self.backbone = nn.Sequential(
			nn.Conv2d(image_c, N // 2, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.05),
			nn.Conv2d(N // 2, N, (5, 5), stride=2, padding=2),
			nn.LeakyReLU(0.05),
			nn.Conv2d(N, N, (5, 5), stride=2, padding=2),
			nn.LeakyReLU(0.05),
			nn.Conv2d(N, 1, (5, 5), stride=2, padding=2),
			nn.LeakyReLU(0.05)
		)
		self.conv_end = nn.Conv2d(N, image_c, (3, 3), stride=1, padding=1)

	def forward(self, x):
		pass