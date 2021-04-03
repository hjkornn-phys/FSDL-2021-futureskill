import torch
import torch.nn as nn
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
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1 **kwargs) # kernel size = ...
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        if not self.activation:
            return self.batchnorm(self.conv(x))
        return self.relu(self.batchnorm(self.conv(x)))

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block,self).__init__()
        if in_channels == out_channels:
            self.convseq = nn.Sequential(
                                        Conv_block(in_channels, out_channels),
                                        Conv_block(out_channels, out_channels, activation=False)
            )
            self.iden = nn.Identity()
        else:
            self.convseq = nn.Sequential(
                                        Conv_block(in_channels, out_channels, stride=2),
                                        Conv_block(out_channels, out_channels, activation=False)
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    
    def forward(self, x):
        y = self.convseq(x)
        x = y + self.iden(x)
        x = nn.ReLU(x)
        return x
