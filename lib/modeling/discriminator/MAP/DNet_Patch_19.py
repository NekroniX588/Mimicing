import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class D_Net(nn.Module):

	def __init__(self, num_of_FM, chennels):
		super(D_Net, self).__init__()
		self.chennels = chennels
		self.num_FM = num_of_FM
		if num_of_FM == 0:
			self.main = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, kernel_size=4, stride=2, padding=1),#9
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels * 2, kernel_size=4, stride=2, padding=1, bias=False),#4
				nn.BatchNorm2d(self.chennels * 2),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels *2, self.chennels * 4, kernel_size=4, stride=2, padding=1, bias=False),#4
				nn.BatchNorm2d(self.chennels * 4),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels * 4, 1, kernel_size=2, bias=False),
				)

		if num_of_FM == 1:
			self.main = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, kernel_size=4, stride=2, padding=1),#9
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels, self.chennels * 2, kernel_size=4, stride=2, padding=1, bias=False),#4
				nn.BatchNorm2d(self.chennels * 2),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels * 2, 1, kernel_size=2, bias=False),
				)

		if num_of_FM == 2:
			self.main = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels * 2, kernel_size=4, stride=2, padding=1, bias=False),#4
				nn.BatchNorm2d(self.chennels * 2),
            	nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(self.chennels * 2, 1, kernel_size=2, bias=False),
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