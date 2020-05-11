'''
--------------------------------------------------------
@File    :   tester.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2020/4/16 19:42     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
''' 
import torch
import sys
class Tester(object):
	""" The class to test networks"""
	def __init__(self, dataloader, network, functions):
		"""
		:param dataloader:  a test dataloader object
		:param network:     a trained network
		:param functions:   a list of functions of which each has two parameters
			to compute test indexes, and the test indexes must be a number.
		"""
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print('device:', self.device)
		self.dataloader = dataloader
		self.net = network.to(self.device)
		self.index_F_list = functions

	def test(self)->list:
		self.net.eval()
		test_indexes = [0] * len(self.index_F_list)
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(self.dataloader):
				data, target = data.to(self.device), target.to(self.device)
				output = self.net(data)
				for i, F in enumerate(self.index_F_list):
					test_indexes[i] += F(output.cpu().numpy(), target.cpu().numpy())
				results = [ele / len(self.dataloader.dataset) for ele in test_indexes]
				sys.stdout.write('\rTest indexes:{}\n'.format(results))
		return results