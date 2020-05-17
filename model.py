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

class Step(nn.Module):

	def __init__(self, N):
		super(Step, self).__init__()
		self.conv_0 = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1)
		self.conv_1 = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1)
		# self.conv_2 = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		x1 = F.leaky_relu(self.conv_0(x), 0.2)
		x2 = F.leaky_relu(self.conv_1(x1), 0.2)
		# x3 = F.leaky_relu(self.conv_2(x2), 0.2) + x1
		return x2

class UNet(nn.Module):
	""" The UNet module"""
	def __init__(self, input_c=3, output_c=3, N=32):
		super(UNet, self).__init__()
		# def the operations in UNet
		self.conv_input = nn.Conv2d(input_c, N, kernel_size=3, stride=1, padding=1)
		self.step_0 = Step(N=N)
		self.down_0 = nn.Conv2d(N, N * 2, kernel_size=3, stride=2, padding=1)
		self.step_1 = Step(N=N * 2)
		self.down_1 = nn.Conv2d(N * 2, N * 4, kernel_size=3, stride=2, padding=1)
		self.step_2 = Step(N=N * 4)
		self.down_2 = nn.Conv2d(N * 4, N * 8, kernel_size=3, stride=2, padding=1)
		self.step_3 = Step(N=N * 8)
		self.down_3 = nn.Conv2d(N * 8, N * 16, kernel_size=3, stride=2, padding=1)
		self.step_4 = Step(N=N * 16)
		self.step_5 = Step(N=N * 16)
		self.up_0 = nn.ConvTranspose2d(N * 16, N * 8, kernel_size=4, stride=2, padding=1)
		self.step_6 = Step(N=N * 8)
		self.up_1 = nn.ConvTranspose2d(N * 8, N * 4, kernel_size=4, stride=2, padding=1)
		self.step_7 = Step(N=N * 4)
		self.up_2 = nn.ConvTranspose2d(N * 4, N * 2, kernel_size=4, stride=2, padding=1)
		self.step_8 = Step(N=N * 2)
		self.up_3 = nn.ConvTranspose2d(N * 2, N, kernel_size=4, stride=2, padding=1)
		self.step_9 = Step(N=N)
		self.conv_output = nn.Conv2d(N, output_c, kernel_size=3, stride=1, padding=1)


	def forward(self, x):
		x_conv_input = F.leaky_relu(self.conv_input(x), 0.2)
		x_step_0 = self.step_0(x_conv_input)
		x_down_0 = F.leaky_relu(self.down_0(x_step_0), 0.2)
		x_step_1 = self.step_1(x_down_0)
		x_down_1 = F.leaky_relu(self.down_1(x_step_1), 0.2)
		x_step_2 = self.step_2(x_down_1)
		x_down_2 = F.leaky_relu(self.down_2(x_step_2), 0.2)
		x_step_3 = self.step_3(x_down_2)
		x_down_3 = F.leaky_relu(self.down_3(x_step_3), 0.2)
		x_step_4 = self.step_4(x_down_3)
		x_step_5 = self.step_5(x_step_4)
		x_up_0 = F.leaky_relu(self.up_0(x_step_5), 0.2)
		x_step_6 = self.step_6(x_step_3 + x_up_0)
		x_up_1 = F.leaky_relu(self.up_1(x_step_6), 0.2)
		x_step_7 = self.step_7(x_step_2 + x_up_1)
		x_up_2 = F.leaky_relu(self.up_2(x_step_7), 0.2)
		x_step_8 = self.step_8(x_step_1 + x_up_2)
		x_up_3 = F.leaky_relu(self.up_3(x_step_8), 0.2)
		x_step_9 = self.step_9(x_step_0 + x_up_3)
		x_conv_output = self.conv_output(x_step_9)
		return x + x_conv_output

#
# class Step(nn.Module):
#
# 	def __init__(self, in_c, out_c, N):
# 		super(Step, self).__init__()
# 		self.conv_0 = nn.Conv2d(in_c, N, kernel_size=3, stride=1, padding=1)
# 		self.conv_1 = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1)
# 		self.conv_2 = nn.Conv2d(N, out_c, kernel_size=3, stride=1, padding=1)
#
# 	def forward(self, x):
# 		x1 = F.leaky_relu(self.conv_0(x), 0.2)
# 		x2 = F.leaky_relu(self.conv_1(x1), 0.2) + x
# 		x3 = F.leaky_relu(self.conv_2(x2), 0.2) + x1
# 		return x3
#

class CNN(nn.Module):
	""" The nueral network for real-time image restoration. """
	def __init__(self, image_c=3, N=64):
		super(CNN, self).__init__()
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