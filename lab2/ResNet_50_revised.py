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

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(Conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) # kernel size = ...
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        if not self.activation:
            return self.batchnorm(self.conv(x))
        return self.relu(self.batchnorm(self.conv(x)))

class Res_block(nn.Module):
    def __init__(self, in_channels, red_channels, out_channels):
        super(Res_block,self).__init__()
        self.relu = nn.ReLU()
        
        if in_channels==64: #ㅇㅋ
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        elif in_channels == out_channels:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Identity()
        else:
            self.convseq = nn.Sequential(
                                    Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                                    Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                                    Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0, stride=2)
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        
    def forward(self, x):
        y = self.convseq(x)
        print(y.shape, x.shape)
        x = y + self.iden(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels=3 , num_classes=1000):
        super(ResNet, self).__init__()
        self.conv1 = Conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2_x = nn.Sequential(
                                        Res_block(64, 64, 256),
                                        Res_block(256, 64, 256),
                                        Res_block(256, 64, 256)
        )
        
        self.conv3_x = nn.Sequential(
                                        Res_block(256, 128, 512),
                                        Res_block(512, 128, 512),
                                        Res_block(512, 128, 512),
                                        Res_block(512, 128, 512)
        )

        self.conv4_x = nn.Sequential(
                                        Res_block(512, 256, 1024),
                                        Res_block(1024, 256, 1024),
                                        Res_block(1024, 256, 1024),
                                        Res_block(1024, 256, 1024),
                                        Res_block(1024, 256, 1024),
                                        Res_block(1024, 256, 1024)
        )
        
        self.conv5_x = nn.Sequential(
                                        Res_block(1024, 512, 2048),
                                        Res_block(2048, 512, 2048),
                                        Res_block(2048, 512, 2048),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = ResNet()
    print(model(x).shape)
