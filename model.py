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
import torch.nn as nn
import torch

class UNet(nn.Module):
	""" The UNet module"""
	def __init__(self, input_c=3, output_c=3, N=32):
		super(UNet, self).__init__()
		# def the operations in UNet
		self.conv_0 = nn.Conv2d(input_c, N, (3, 3), stride=1, padding=1)
		self.relu_0 = nn.LeakyReLU(0.2)
		self.U = nn.Sequential(
			# step_0
			nn.Conv2d(N, N, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N, N, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			# step_1
			nn.Conv2d(N, N * 2, (3, 3), stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 2, N * 2, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 2, N * 2, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			# step_2
			nn.Conv2d(N * 2, N * 4, (3, 3), stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 4, N * 4, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 4, N * 4, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			# step_3
			nn.Conv2d(N * 4, N * 8, (3, 3), stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 8, N * 8, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 8, N * 8, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			# step_4
			nn.Conv2d(N * 8, N * 16, (3, 3), stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 16, N * 16, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 16, N * 16, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 16, N * 16, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 16, N * 16, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 16, N * 16, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			# step_3
			nn.ConvTranspose2d(N * 16, N * 8, (3, 3), stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 8, N * 8, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 8, N * 8, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			# step_2
			nn.ConvTranspose2d(N * 8, N * 4, (3, 3), stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 4, N * 4, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 4, N * 4, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			# step_1
			nn.ConvTranspose2d(N * 4, N * 2, (3, 3), stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 2, N * 2, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N * 2, N * 2, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			# step_0
			nn.ConvTranspose2d(N * 2, N, (3, 3), stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N, N, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(N, N, (3, 3), stride=1, padding=1),
			nn.LeakyReLU(0.2),
		)
		self.conv_end = nn.Conv2d(N, output_c, (3, 3), stride=1, padding=1)

	def forward(self, x):
		F_0 = self.relu_0(self.conv_0(x))
		F_U = self.U(F_0)
		F_end = self.conv_end(F_U)
		return F_end



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