import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.modeling.discriminator.attention import AttentionConv
# from attention import AttentionConv

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, groups=1, base_width=64, bias=False):
        super(Bottleneck, self).__init__()
        self.stride = stride
        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=bias),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            AttentionConv(width, width, kernel_size=7, padding=3, groups=8),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, stride = stride, bias=bias),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class D_Net(nn.Module):

	def __init__(self, num_of_FM, chennels):
		super(D_Net, self).__init__()
		self.chennels = chennels
		self.num_FM = num_of_FM
		self.Sigmoid = nn.Sigmoid()

		if num_of_FM == 0:
			self.features = []
			self.features.append(Bottleneck(self.chennels, self.chennels//2, 1, bias=True))
			self.features.append(Bottleneck(self.chennels, self.chennels//4, 2, bias=True))
			self.features.append(Bottleneck(self.chennels//2, self.chennels//8, 2, bias=True))
			self.features.append(Bottleneck(self.chennels//4, self.chennels//8, 2, bias=True))
			self.features.append(nn.Conv2d(self.chennels//4, 1, kernel_size=3, bias=True))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 1:
			self.features = []
			self.features.append(Bottleneck(self.chennels, self.chennels//4, 1, bias=True))
			self.features.append(Bottleneck(self.chennels//2, self.chennels//8, 2, bias=True))
			self.features.append(Bottleneck(self.chennels//4, self.chennels//8, 2, bias=True))
			self.features.append(nn.Conv2d(self.chennels//4, 1, kernel_size=3, bias=True))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 2:
			self.features = []
			self.features.append(Bottleneck(self.chennels, self.chennels//2, 1, bias=True))
			self.features.append(Bottleneck(self.chennels, self.chennels//4, 2, bias=True))
			self.features.append(nn.Conv2d(self.chennels//2, 1, kernel_size=3, bias=True))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 3:
			self.features = []
			self.features.append(Bottleneck(self.chennels, self.chennels//2, 1, bias=True))
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=3, bias=True))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 4:
			
			self.features = []
			self.features.append(Bottleneck(self.chennels, self.chennels//2, 1, bias=True))
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=2,  bias=False))
			self.main = nn.Sequential(*self.features)	

			
		if num_of_FM == 5:
			block = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, 1, 1, 0, bias=False),
				nn.BatchNorm2d(self.chennels),
				nn.ReLU(inplace=True),
				# dw
				nn.Conv2d(self.chennels, self.chennels, 1, groups=self.chennels//2, bias=False),
				nn.BatchNorm2d(self.chennels),
				# pw-linear
				nn.Conv2d(self.chennels, self.chennels, 1, 1, 0, bias=False),
				nn.BatchNorm2d(self.chennels),
				nn.ReLU(inplace=True))

			self.features = []
			for i in range(2): 
				self.features.append(block) 

			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=1,  bias=False))
			self.main = nn.Sequential(*self.features)	


	def forward(self, x):
		x = self.main(x)
		x = x.view(-1)
		x = self.Sigmoid(x)
		return x

if __name__ == '__main__':
	CHANALS_MAP = [512, 1024, 512, 256, 256, 256]
	D_Nets = []
	x = torch.rand(32,512,5,5)
	NET = D_Net(2, 512)
	print(NET)
	print(NET(x).size())

