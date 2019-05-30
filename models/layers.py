import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

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

class ResNet18Begin(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x

class resnet34(nn.Module):
    """Constructs a ResNet-34 model.
    Returns a resnet model optionally pre-trained on ImageNet
    """
    def __init__(self, pretrained=False):
        super().__init__()
        original_model = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x

class Branch(nn.Module):
    def __init__(self, branched_submodules):
        super().__init__()
        self.branched_submodules = nn.ModuleList(branched_submodules)
    def forward(self, x):
        output_branches = [branch(x) for branch in self.branched_submodules]
        return output_branches


