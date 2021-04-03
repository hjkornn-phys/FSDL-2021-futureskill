import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, architecture = VGG16):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(architecture)
        
        self.fcs = nn.Sequential(
            nn.Linear(architecture[-2]*7*7, 4096), # 224/2**5 (5 maxpool)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1) # Batch Size
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                # Core
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(x),
                                    nn.ReLU()]

                in_channels = x
            
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)

if __name__ == '__main__':
    model = VGG_net()
    X = torch.randn(1, 3, 224, 224)
    print(model(X).shape)