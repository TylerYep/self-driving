import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FC(nn.Module):
    def __init__(self, input_size, output_size, drop_prob):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(p=drop_prob)
        self.FC = nn.Linear(input_size, output_size)

    def forward(self, x) -> torch.Tensor:
        x_out = self.FC(x)
        relu_x_out = F.relu(x_out)
        out = self.dropout(relu_x_out)
        return out

class Conv2dReluDropout(nn.Module):
    def __init__(self, filter_size, pad, stride, num_filters, in_channels, drop_prob):
        super().__init__()
        self.filter_size = filter_size
        self.pad = pad
        self.stride = stride
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, num_filters, filter_size, stride, pad)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x) -> torch.Tensor:
        x_convout = self.conv(x)
        relu_x_convout = F.relu(x_convout)
        out = self.dropout(relu_x_convout)
        return out