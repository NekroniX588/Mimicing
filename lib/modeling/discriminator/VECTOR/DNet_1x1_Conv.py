import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class D_Net(nn.Module):

	def __init__(self, num_of_FM, chennels):
		super(D_Net, self).__init__()
		self.chennels = chennels
		self.num_FM = num_of_FM

		self.main_1 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size=self.chennels),#10
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.main_2 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size=self.chennels//2),#10
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.main_3 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size=self.chennels//4),#10
			nn.LeakyReLU(0.2, inplace=True),
		)
		size = 1 + (self.chennels - self.chennels//2 + 1) + (self.chennels - self.chennels//4 + 1)
		self.end = nn.Sequential(
			nn.Conv1d(4, 1, kernel_size=size),#10
			nn.LeakyReLU(0.2, inplace=True),
			)


	def forward(self, x):
		x = x.unsqueeze(1)
		x_1 = self.main_1(x)
		# print(x_1.size())
		x_2 = self.main_2(x)
		# print(x_2.size())
		x_3 = self.main_3(x)
		# print(x_3.size())
		x = torch.cat([x_1, x_2, x_3],dim=2)
		x = self.end(x)
		x = x.view(-1)
		x = F.sigmoid(x)
		return x

if __name__ == '__main__':
	writer = SummaryWriter('./')
	CHANALS_MAP = [512, 1024, 512, 256, 256, 256]
	D_Nets = []
	for i,v in enumerate(CHANALS_MAP):
		D_Nets.append(D_Net(i, v))
	for i in range(len(D_Nets)):
		x = torch.rand(32,1,CHANALS_MAP[i])
		print(D_Nets[i](x))
		writer.add_scalar('result', D_Nets[i](x)[0])
		print(D_Nets[i](x).size())
