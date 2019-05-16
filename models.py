import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NaiveConditionedCNN(nn.Module):
    '''
    Simple conditioned CNN:
    RGB Image as input to CNN, high-level direction and speed as input to a feed-forward network
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dReluDropout(filter_size=5, pad=0, stride=2, num_filters=24, in_channels=3, drop_prob=0.2)
        self.conv2 = Conv2dReluDropout(filter_size=5, pad=0, stride=2, num_filters=36, in_channels=24, drop_prob=0.2)
        self.conv3 = Conv2dReluDropout(filter_size=5, pad=0, stride=2, num_filters=48, in_channels=36, drop_prob=0.2)
        self.conv4 = Conv2dReluDropout(filter_size=3, pad=0, stride=1, num_filters=64, in_channels=48, drop_prob=0.2)
        self.conv5 = Conv2dReluDropout(filter_size=3, pad=0, stride=1, num_filters=64, in_channels=64, drop_prob=0.2)

        self.FC1 = FC(input_size=64*1*18, output_size=100, drop_prob=0.5) # See output of self.conv5
        self.FC2 = FC(input_size=100, output_size=50, drop_prob=0.5)
        self.FC3 = nn.Linear(50, 10)

        # concatenate the output of FC3 and cond_FC1
        self.FC4 = nn.Linear(12, 1)

        # Feed-forward layers for conditioned features: speed, 1-hot (3-dim) high level control
        self.cond_FC1 = FC(input_size=4, output_size=2, drop_prob=0.0)


    def forward(self, x, measurements) -> torch.Tensor:
        '''
        param x: shape (N, 66, 200, 3) (N, H, W, C)
        '''
        N, H, W, C = x.shape
        x = x.reshape((N, C, H, W))
        # We omit the normalization layer proposed in Bojarski et al.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) # output shape (N, 64, 3, 20)
        x = self.conv5(x) # output shape (N, 64, 1, 18)

        # Flatten layer before FC layers
        x = x.reshape((N, -1))
        measurements = measurements.reshape(N, -1)

        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        x = F.relu(x)

        measurements = self.cond_FC1(measurements)
        cat_x = torch.cat((x, measurements), 1)

        x = self.FC4(cat_x)
        return x


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


if __name__ == '__main__':
    data = torch.from_numpy(np.zeros((32, 66, 200, 3))).float()
    cond_features = torch.zeros((32, 4, 1))
    model = NaiveConditionedCNN()
    out = model(data, cond_features)
    print(out.shape)
