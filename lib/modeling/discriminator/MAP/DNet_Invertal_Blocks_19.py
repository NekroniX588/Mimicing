import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)



class D_Net(nn.Module):

	def __init__(self, num_of_FM, chennels):
		super(D_Net, self).__init__()
		self.chennels = chennels
		self.num_FM = num_of_FM
		self.Sigmoid = nn.Sigmoid()
		if num_of_FM == 0:
			self.features = []
			for i in range(2): #19
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1)) 
			self.features.append(InvertedResidual(self.chennels, self.chennels, 2, 2)) 

			for i in range(2): #10
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1)) 
			self.features.append(InvertedResidual(self.chennels, self.chennels, 2, 2))

			for i in range(2): #5
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1))
			self.features.append(InvertedResidual(self.chennels, self.chennels, 2, 2)) 

			for i in range(2): #3
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1))
			
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 1:
			self.features = []
			for i in range(3): #10
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1)) 
			self.features.append(InvertedResidual(self.chennels, self.chennels, 2, 2)) 

			for i in range(3): #5
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1)) 
			self.features.append(InvertedResidual(self.chennels, self.chennels, 2, 2))

			for i in range(2): #3
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1))
			
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 2:
			self.features = []
			for i in range(4): #5
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1)) 
			self.features.append(InvertedResidual(self.chennels, self.chennels, 2, 2)) 

			for i in range(4): #3
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1)) 
			
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 3:
			self.features = []
			for i in range(8): #3
				self.features.append(InvertedResidual(self.chennels, self.chennels, 1, 1)) 
			
			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=3, bias=False))
			self.main = nn.Sequential(*self.features)

		if num_of_FM == 4:
			block = nn.Sequential(
				nn.Conv2d(self.chennels, self.chennels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.chennels),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(self.chennels, self.chennels, 2, stride=2, padding=1, groups=self.chennels//2, bias=False),
                nn.BatchNorm2d(self.chennels),
                # pw-linear
                nn.Conv2d(self.chennels, self.chennels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.chennels),
                nn.ReLU(inplace=True))
			
			self.features = []
			for i in range(8): #3
				self.features.append(block) 
			
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
			for i in range(4): 
				self.features.append(block) 

			self.features.append(nn.Conv2d(self.chennels, 1, kernel_size=1,  bias=False))
			self.main = nn.Sequential(*self.features)	


	def forward(self, x):
		x = self.main(x)
		x = x.view(-1)
		x = self.Sigmoid(x)
		return x
