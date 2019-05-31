import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from .layers import FC, ResNet18Begin, resnet34, Branch

class BranchedCOIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet34 = resnet34(pretrained=True)
        num_branches = 3
        branch_fc_list = []
        for i in range(num_branches):
            branch = nn.Sequential(FC(512, 256, drop_prob=0.0),
                                   FC(256, 256, drop_prob=0.5),
                                   nn.Linear(256, 2))
            branch_fc_list.append(branch)
        self.branches = Branch(branch_fc_list)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, measurements):
        N, H, W, C = x.shape
        x = x.reshape((N, C, H, W))
        x = self.resnet34(x)
        x = x.reshape(N, -1)
        output_branches = self.branches(x)
        return output_branches

if __name__ == '__main__':
    data = torch.zeros((32, 224, 600, 3)).float()
    cond_features = torch.zeros((32, 4, 1))
    model = BranchedCOIL()
    out = model(data, cond_features)
    print(out.shape)
