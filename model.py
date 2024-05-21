import torch.nn as nn
import numpy as np
import torch
import math

class MyModel(nn.Module):
	def __init__(self, nNodes):
		super(MyModel, self).__init__()
		self.weight=nn.Parameter(torch.FloatTensor(nNodes, nNodes), requires_grad=True)				
		self.reset_parameters()
	
	def reset_parameters(self, weight_initial=None):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(0, stdv)
	
	def forward1(self, x, tau):
		NewX=torch.abs((1.0-2.*x))*tau*tau/2.0+x*(1.0-x)*tau
		NewX*=(1.0-x)
		gain=torch.matmul(NewX, self.weight)	
		return torch.clamp(x+gain, 0.0, 1.0)

	def forward(self, x, tau):				
		NeighborRate=torch.matmul(x, self.weight)	
		DiffRate=(1.0-x)*NeighborRate					
		return torch.clamp(x+tau*DiffRate, 0.0, 1.0)