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


from dataset import TrainDataset, EvalDataset
from model import UNet, CNN
from trainer import Trainer
import torch.nn.functional as F
import sys, torch, os.path

model_name = sys.argv[1]
batch_size = int(sys.argv[2])
pkls_dir = sys.argv[3]
dev = sys.argv[4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() and dev=='cuda' else "cpu")
print(model_name, batch_size, pkls_dir, DEVICE)

if model_name == 'UNet-16':
	model = UNet(N=16)
elif model_name == 'UNet-32':
	model = UNet(N=32)
elif model_name == 'UNet-64':
	model = UNet(N=64)
elif model_name == 'CNN':
	model = CNN(N=32)
else:
	model = None
	print('>>> Model name:{} invalid!')
	exit(-1)
# run "python train.py ./pkls UNet-16"

if __name__ == '__main__':
	# load train data
	train_dataset = TrainDataset(datadir='data/Train/Toled')
	eval_dataset = EvalDataset(datadir='data/Eval/Toled', npy=True)
	# create model for Image-Restoration
	trainer = Trainer(train_dataset=train_dataset,
	                  eval_dataset=eval_dataset,
	                  network=model,
	                  loss_function=F.l1_loss,
	                  batch_size=batch_size,
	                  learning_rate=1e-4,
	                  epoch=400,
	                  pkls_dir=os.path.join(pkls_dir, model_name),
	                  summary_dir='./summarylogs/{}'.format(model_name))
	model = trainer.train()