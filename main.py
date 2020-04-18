'''
--------------------------------------------------------
@File    :   main.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2020/4/16 17:18     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
'''
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import PairedData
from torchvision import datasets, transforms
from trainer import Trainer
from tester import Tester


class ConvNet(nn.Module):
	def __init__(self):
		super().__init__()
		# batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
		# 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
		self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
		self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
		# 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
		self.fc1 = nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
		self.fc2 = nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
	def forward(self,x):
		in_size = x.size(0) # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
		out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
		out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
		out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
		out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
		out = F.relu(out) # batch*20*10*10
		out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
		out = self.fc1(out) # batch*2000 -> batch*500
		out = F.relu(out) # batch*500
		out = self.fc2(out) # batch*500 -> batch*10
		out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
		return out

def acc(y_, y):
	index_val = 0
	pred = y_.max(1, keepdim=True)[1]  # 找到概率最大的下标
	index_val += pred.eq(y.view_as(pred)).sum().item()
	return index_val
	# print('acc:', index_val)

def MNIST_Test():
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data', train=True, download=True,
		               transform=transforms.Compose([
			               transforms.ToTensor(),
			               transforms.Normalize((0.1307,), (0.3081,))
		               ])),
		batch_size=32, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data', train=False, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])),
		batch_size=32, shuffle=True)
	model = ConvNet()
	trainer = Trainer(dataloader=train_loader,
	                  network=model,
	                  loss_function=F.nll_loss,
	                  epoch=10)
	model = trainer.train()
	tester = Tester(test_loader, model, [acc])
	tester.test()

if __name__ == '__main__':
	train_loader = DataLoader(PairedData(), batch_size=8, shuffle=True)
	model = ConvNet()
	trainer = Trainer(dataloader=train_loader,
	                  network=model,
	                  loss_function=F.nll_loss,
	                  epoch=10)
	model = trainer.train()