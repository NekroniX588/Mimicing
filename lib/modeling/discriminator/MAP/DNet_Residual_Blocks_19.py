import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
	

class BasicBlock(nn.Module):
	expansion = 1
	__constants__ = ['downsample']

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 2

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.stride = stride
		if self.stride != 1:
			self.downsample = nn.Sequential(conv1x1(inplanes, planes * self.expansion, self.stride),
										norm_layer(planes * self.expansion),)
		else:
			self.downsample = None


	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck_2(nn.Module):
	expansion = 2

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck_2, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = nn.Conv2d(width, width, kernel_size=2 ,stride=2, padding=1)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.stride = stride
		if self.stride != 1:
			self.downsample = nn.Sequential(conv1x1(inplanes, planes * self.expansion, self.stride),
										norm_layer(planes * self.expansion),)
		else:
			self.downsample = None

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out



class D_Net(nn.Module):

	def __init__(self, num_of_FM, chennels):
		super(D_Net, self).__init__()
		self.chennels = chennels
		self.num_FM = num_of_FM
		self.Sigmoid = nn.Sigmoid()

		if num_of_FM == 0:
			self.features = []
			for i in range(2): #19
				self.features.append(Bottleneck(self.chennels, self.chennels//2)) 
			self.features.append(Bottleneck(self.chennels, self.chennels, 2)) 

			self.chennels = self.chennels * 2

			for i in range(1): #19
				self.features.append(Bottleneck(self.chennels, self.chennels//2)) 
			self.features.append(Bottleneck(self.chennels, self.chennels//2, 2)) 

			for i in range(1): #19
				self.features.append(Bottleneck(self.chennels, self.chennels//2)) 
			self.features.append(Bottleneck(self.chennels, self.chennels//2, 2)) 

			for i in range(1): #19
				self.features.append(Bottleneck(self.chennels, self.chennels//2)) 
			
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 1:
			self.features = []
			for i in range(2): #19
				self.features.append(Bottleneck(self.chennels, self.chennels//2)) 
			self.features.append(Bottleneck(self.chennels, self.chennels, 2)) 

			self.chennels = self.chennels * 2

			for i in range(1): #5
				self.features.append(Bottleneck(self.chennels, self.chennels//2)) 
			self.features.append(Bottleneck(self.chennels, self.chennels//2, 2)) 

			for i in range(1): #3
				self.features.append(Bottleneck(self.chennels, self.chennels//2)) 
			
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 2:
			self.features = []
			for i in range(2): #5
				self.features.append(Bottleneck(self.chennels, self.chennels//2)) 
			self.features.append(Bottleneck(self.chennels, self.chennels, 2)) 

			self.chennels = self.chennels * 2

			for i in range(2): #3
				self.features.append(Bottleneck(self.chennels, self.chennels//2))  
			
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 3:
			self.features = []
			for i in range(3): #3
				self.features.append(Bottleneck(self.chennels, self.chennels//2)) 
			
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 4:
			
			self.features = []
			for i in range(3): #3
				self.features.append(Bottleneck_2(self.chennels, self.chennels//2)) 
			
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
