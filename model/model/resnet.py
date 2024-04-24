import torch
import torch.nn as nn
from typing import Type, Union, List, Optional 


# 3x3卷积
def conv3x3(in_channels, out_channels, stride=1, initial_zero=False):

    bn = nn.BatchNorm2d(out_channels)
    if initial_zero == True:
        nn.init.constant_(bn.weight, 0)

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
    )

def conv1x1(in_channels, out_channels, stride=1, initial_zero=False):
    bn = nn.BatchNorm2d(out_channels)
    if initial_zero == True:
        nn.init.constant_(bn.weight, 0)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False),
    )


# ResidualUnit 残差单元
class ResidualUnit(nn.Module):
    def __init__(self, out_channels: int, stride_one: int = 1, in_channels: Optional[int] = None):
        super().__init__()

        self.stride_one = stride_one

        if stride_one != 1:
            in_channels = int(out_channels / 2)
        else:
            in_channels = out_channels

        self.fit_ = nn.Sequential(
            conv3x3(in_channels, out_channels, stride=stride_one),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels, initial_zero=True),
        )
        self.skip_conv = conv1x1(in_channels, out_channels, stride=stride_one)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.fit_(x)
        x = x
        if self.stride_one == 2:
            x = self.skip_conv(x)
        hx = self.relu(x + fx)
        return hx


# 瓶颈
class BottleNeck(nn.Module):
    def __init__(self, middle_channels, stride_one=1, in_channels: Optional[int] = None):
        super().__init__()
        out_channels = middle_channels * 4

        if in_channels == None:

            if stride_one != 1:

                in_channels = 2 * middle_channels
            else:

                in_channels = 4 * middle_channels

        self.fit_ = nn.Sequential(
            conv1x1(in_channels, middle_channels, stride=stride_one),
            nn.ReLU(inplace=True),
            conv3x3(middle_channels, middle_channels),
            nn.ReLU(inplace=True),
            conv1x1(middle_channels, out_channels),
        )
        self.skip_conv = conv1x1(in_channels, out_channels, stride_one)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.fit_(x)
        x = self.skip_conv(x)
        hx = self.relu(x + fx)
        return hx

def make_layers(
        block: Type[Union[ResidualUnit, BottleNeck]],
        middle_channels: int,
        num_blocks: int,
        afterconv1: bool = False
):
    layers = []
    if afterconv1:
        layers.append(block(middle_channels, in_channels=64))
    else:
        layers.append(block(middle_channels, stride_one=2))
    for i in range(num_blocks - 1):
        layers.append(block(middle_channels))
    return layers


# 复现ResNet
class ResNet(nn.Module):
    def __init__(self, block: Type[Union[ResidualUnit, BottleNeck]], layers_blocks_num: list[int], num_classes: int,
                 lr: int, epochs: int, local_epochs: int, bs: int, gamma: int):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )
        self.layer2 = nn.Sequential(*make_layers(block, 64, layers_blocks_num[0], afterconv1=True))
        self.layer3 = nn.Sequential(*make_layers(block, 128, layers_blocks_num[1]))
        self.layer4 = nn.Sequential(*make_layers(block, 256, layers_blocks_num[2]))
        self.layer5 = nn.Sequential(*make_layers(block, 512, layers_blocks_num[3]))
  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
  
        if block == ResidualUnit:
            self.fc = nn.Linear(512, num_classes)
        else:
            self.fc = nn.Linear(2048, num_classes)

        self.lr = lr
        self.epochs = epochs
        self.local_epochs = local_epochs
        self.bs = bs
        self.gamma = gamma

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer5(self.layer4(self.layer3(self.layer2(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
