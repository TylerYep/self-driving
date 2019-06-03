import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .layers import FC, ResNet34, Branch, Conv2dReluDropout, ResNet18

class BranchedNvidia(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dReluDropout(filter_size=5, pad=0, stride=2, num_filters=24, in_channels=3, drop_prob=0.2)
        self.conv2 = Conv2dReluDropout(filter_size=5, pad=0, stride=2, num_filters=36, in_channels=24, drop_prob=0.2)
        self.conv3 = Conv2dReluDropout(filter_size=5, pad=0, stride=2, num_filters=48, in_channels=36, drop_prob=0.2)
        self.conv4 = Conv2dReluDropout(filter_size=3, pad=0, stride=1, num_filters=64, in_channels=48, drop_prob=0.2)
        self.conv5 = Conv2dReluDropout(filter_size=3, pad=0, stride=1, num_filters=64, in_channels=64, drop_prob=0.2)

        num_branches = 3
        branch_fc_list = []
        for i in range(num_branches):
            branch = nn.Sequential(FC(1152, 100, drop_prob=0.5),
                                   FC(100, 50, drop_prob=0.5),
                                   FC(50, 10, drop_prob=0.5),
                                   nn.Linear(10, 2))
            branch_fc_list.append(branch)
        self.branches = Branch(branch_fc_list)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.constant_(m.bias, 0.1)

    def forward(self, x, measurements):
        N, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)

        # We omit the normalization layer proposed in Bojarski et al.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # output shape (N, 64, 3, 20)
        x = self.conv5(x) # output shape (N, 64, 1, 18)

        # Flatten layer before FC layers
        x = x.reshape(N, -1)
        output_branches = self.branches(x)
        return output_branches

    def forward_with_activations(self, x, measurements):
        N, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)

        # We omit the normalization layer proposed in Bojarski et al.
        first_activation = self.conv1(x)
        second_activation = self.conv2(first_activation)
        third_activation = self.conv3(second_activation)
        x = self.conv4(third_activation) # output shape (N, 64, 3, 20)
        x = self.conv5(x) # output shape (N, 64, 1, 18)

        # Flatten layer before FC layers
        x = x.reshape(N, -1)
        output_branches = self.branches(x)
        return output_branches, [first_activation, second_activation, third_activation]


class BranchedCOIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet34 = ResNet34(pretrained=True)
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
        x = x.permute(0, 3, 1, 2)
        x = self.resnet34(x)
        x = x.reshape(N, -1)
        output_branches = self.branches(x)
        return output_branches

class BranchedCOIL_ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = ResNet18(pretrained=True)
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
        x = x.permute((0, 3, 1, 2))
        x = self.resnet18(x)
        x = x.reshape(N, -1)
        output_branches = self.branches(x)
        return output_branches


if __name__ == '__main__':
    data = torch.zeros((32, 66, 200, 3)).float()
    cond_features = torch.zeros((32, 4, 1))
    model = BranchedCOIL_ResNet18()
    out = model(data, cond_features)
    #print(out)
