import torch
import torch.nn as nn
import torch.nn.functional as F

class NaiveConditionedCNN(nn.Module):
	''' Simple conditioned CNN:
	RGB Image as input to CNN, high-level direction and speed as input to a feed-forward network
	'''
	def __init__(self):
		super(NaiveConditionedCNN, self).__init__()

	def forward(self, input) -> torch.Tensor:



class Conv2dReluDropout(nn.Module):
	def __init__(self, filter_size, pad, stride, num_filters):

	def forward(self, input) -> torch.Tensor: