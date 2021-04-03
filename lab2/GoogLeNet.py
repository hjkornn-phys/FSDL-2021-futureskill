import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet,self).__init__()

        self.conv1 = Conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # In this order in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.incention3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.incention4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.incention4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.incention4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.incention4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.incention4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.incention3b(x)
        x = self.maxpool3(x)

        x = self.incention4a(x)
        x = self.incention4b(x)
        x = self.incention4c(x)
        x = self.incention4d(x)
        x = self.incention4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # BATCH
        x = self.dropout(x)
        x = self.fc(x)
        return x 


        
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

if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = GoogLeNet()
    print(model(x).shape)