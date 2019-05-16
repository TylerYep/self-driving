import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms

class PretrainedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.resnet18 = models.resnet18(pretrained=True)

    def forward(self, x, measurements):
        '''
        param x: shape (N, 224, 600, 3) (N, H, W, C)
        '''
        N, H, W, C = x.shape
        x = x.reshape((N, C, H, W))
        print(x.shape)
        x[0] = self.normalize(x[0]) # Need to use normalize on all examples in the batch
        out = self.resnet18(x)
        return out

if __name__ == '__main__':
    data = torch.from_numpy(np.zeros((32, 224, 600, 3))).float()
    cond_features = torch.zeros((32, 4, 1))
    model = PretrainedResNet()
    out = model(data, cond_features)
    print(out.shape)