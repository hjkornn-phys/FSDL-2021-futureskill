import torch
import torch.nn as nn
from torch.nn.modules import activation
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import math

class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool)  -> None:
        super(Inception_block, self).__init__()
        self.branch1 = Conv_block(in_channels, out_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential( 
                                    Conv_block(in_channels, red_3x3, kernel_size=1),
                                    Conv_block(red_3x3, out_3x3, kernel_size =3, padding=1)
        )

        self.branch3 = nn.Sequential(
                                    Conv_block(in_channels, red_5x5, kernel_size=1),
                                    Conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
                                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                    Conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self,x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(Conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) # kernel size = ...
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class SE_block(nn.Module):
    def __init__(self, in_channels, red_ratio): # 입력 채널과 압축률을 설정합니다
        super(SE_block, self).__init__()
        self.fc1 = nn.Linear(in_channels, math.floor(in_channels/red_ratio))
        self.fc2 = nn.Linear(math.floor(in_channels/red_ratio), in_channels)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self,x):
        # Squeeze
        y = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]) , stride=1)(x)
        # Exitation
        y = y.reshape(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigm(y)
        # reshape y
        y = y.reshape(y.shape[0], y.shape[1], 1, 1)
        return x * y

class SE_Inception(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super().__init__()
        self.incep = Inception_block(in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool)
        self.SE = SE_block(out_1x1+out_3x3+out_5x5+out_1x1pool, 3)

    def forward(self,x):
        x = self.SE(self.incep(x))
        return x




if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = SE_Inception(3, 1, 1, 2, 1, 2, 2) # Inception block의 각 가지당 채널 수 , in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    print(model(x).shape)





