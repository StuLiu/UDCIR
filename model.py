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

class Restorer(nn.Module):
	""" The nueral network for real-time image restoration. """
	def __init__(self, image_c=3, N=64):
		super(Restorer, self).__init__()
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

class BaseModel(nn.Module):
	def __init__(self, batch_size, image_h, image_w, image_c, scale_rate):
		super(BaseModel, self).__init__()
		self.batch_size = batch_size
		self.image_h = image_h
		self.image_w = image_w
		self.image_c = image_c
		self.scale_rate = scale_rate

class ESPCN(BaseModel):
	"""
	Real-Time Single Image and Video Super-Resolution Using
	an Efficient Sub-Pixel Convolutional Neural Network
	"""
	def __init__(self, batch_size, image_h, image_w, image_c=3, scale_rate=4):
		assert type(scale_rate) == type(int(1))
		super(ESPCN, self).__init__(batch_size, image_h, image_w, image_c, scale_rate)
		# def the operations
		self.conv_1 = nn.Conv2d(image_c, 64, (5, 5), stride=1, padding=2)
		self.conv_2 = nn.Conv2d(64, 32, (3, 3), stride=1, padding=1)
		self.conv_3 = nn.Conv2d(32, scale_rate**2 * image_c, (3, 3), stride=1, padding=1)
		self.subpixel_conv = nn.PixelShuffle(scale_rate)
		self.act = nn.LeakyReLU()

	def forward(self, x):
		# print('x.shape', x.shape)
		map_1 = self.act(self.conv_1(x))        # torch.Size([bs, 64, image_h, image_w])
		# print('map_1.shape', map_1.shape)
		map_2 = self.act(self.conv_2(map_1))    # torch.Size([bs, 32, image_h, image_w])
		# print('map_2.shape', map_2.shape)
		map_3 = self.act(self.conv_3(map_2))    # torch.Size([bs, 4, image_h, image_w])
		# print('map_3.shape', map_3.shape)
		out = self.subpixel_conv(map_3)         # torch.Size([bs, image_c, image_h*s, image_w*s])
		# print('out.shape', out.shape)
		return out