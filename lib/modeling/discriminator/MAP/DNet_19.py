import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

class D_Net(nn.Module):

	def __init__(self, num_of_FM, chennels):
		super(D_Net, self).__init__()
		self.chennels = chennels
		self.num_FM = num_of_FM
		if num_of_FM == 0:
			self.main = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, kernel_size=3, padding=1),#19
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels, kernel_size=3, stride=2, padding=1),#10
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels, kernel_size=3, stride=2, padding=1),#5
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels, kernel_size=3),
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False),
				)

		if num_of_FM == 1:
			self.main = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, kernel_size=3, padding=1),#10
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels, kernel_size=3, stride=2, padding=1),#5
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels, kernel_size=3),
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False),
				)
		if num_of_FM == 2:
			self.main = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, kernel_size=3, padding=1),
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels, kernel_size=3),
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, 1, kernel_size=3),
				)
		if num_of_FM == 3:
			self.main = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, kernel_size=3, padding=1),#3
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels, kernel_size=3, padding=1),#3
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False),
				)
		if num_of_FM == 4:
			self.main = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, kernel_size=2, stride=2, padding=1),
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels, kernel_size=2, stride=2, padding=1),
				nn.InstanceNorm2d(self.chennels),
            	nn.LeakyReLU(0.2, inplace=True),
            	
				nn.Conv2d(self.chennels, 1, kernel_size=2, bias=False),
				)
		if num_of_FM == 5:
			self.main = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, kernel_size=1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(self.chennels, self.chennels, kernel_size=1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(self.chennels, 1, kernel_size=1, bias=False),
				)

	def forward(self, x):
		x = self.main(x)
		x = x.view(-1)
		x = F.sigmoid(x)
		return x

if __name__ == '__main__':
	CHANALS_MAP = [512, 1024, 512, 256, 256, 256]
	writer = SummaryWriter(log_dir='./')
	for k, maps in enumerate(CHANALS_MAP):
		net = D_Net(k, maps)
		writer.add_graph(D_Net, torch.rand(1,maps,19,19))
	