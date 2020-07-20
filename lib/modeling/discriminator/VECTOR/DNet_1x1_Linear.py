import torch
import torch.nn as nn
import torch.nn.functional as F

class D_Net(nn.Module):

	def __init__(self, num_of_FM, chennels):
		super(D_Net, self).__init__()
		self.chennels = chennels
		self.num_FM = num_of_FM
		self.main = nn.Sequential(
			nn.Linear(self.chennels, self.chennels),#10
			nn.ReLU(),
			nn.Linear(self.chennels, self.chennels//2),
			nn.ReLU(),
			nn.Linear(self.chennels//2, 1),
			)

	def forward(self, x):
		x = self.main(x)
		x = x.view(-1)
		x = F.sigmoid(x)
		return x

if __name__ == '__main__':
	CHANALS_MAP = [512, 1024, 512, 256, 256, 256]
	D_Nets = []
	for i,v in enumerate(CHANALS_MAP):
		D_Nets.append(D_Net(i, v))
	for i in range(len(D_Nets)):
		x = torch.rand(32,CHANALS_MAP[i])
		print(D_Nets[i](x))
		print(D_Nets[i](x).size())
