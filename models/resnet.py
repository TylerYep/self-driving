import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from .layers import FC, ResNet18Begin

class PretrainedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)

        self.resnet18_features = ResNet18Begin(resnet18)
        '''for param in self.resnet18_features.parameters():
            param.requires_grad = False'''
        # Currently runs a feed forward network after frozen resnet feature extractor
        self.FC0 = FC(input_size = 10752, output_size=1000, drop_prob=0.1)

        self.FC1 = FC(input_size=1000, output_size=100, drop_prob=0.1) # See output of self.conv5
        self.FC2 = FC(input_size=100, output_size=50, drop_prob=0.1)
        self.FC3 = nn.Linear(50, 10)

        # concatenate the output of FC3 and cond_FC1
        self.FC4 = nn.Linear(12, 2)

        # Feed-forward layers for conditioned features: speed, 1-hot (3-dim) high level control
        self.cond_FC1 = FC(input_size=4, output_size=2, drop_prob=0.0)

    def forward(self, x, measurements):
        '''
        param x: shape (N, 224, 600, 3) (N, H, W, C) (normalized)
        '''
        N, H, W, C = x.shape
        x = x.reshape((N, C, H, W))
        x = self.resnet18_features(x)
        x = x.reshape(N, -1)
        x = self.FC0(x)

        # Flatten layer before FC layers
        measurements = measurements.reshape(N, -1)

        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x = F.relu(x)

        measurements = self.cond_FC1(measurements)
        cat_x = torch.cat((x, measurements), 1)

        x = self.FC4(cat_x)
        return x

if __name__ == '__main__':
    data = torch.from_numpy(np.zeros((32, 224, 600, 3))).float()
    cond_features = torch.zeros((32, 4, 1))
    model = PretrainedResNet()
    out = model(data, cond_features)
    print(out.shape)
