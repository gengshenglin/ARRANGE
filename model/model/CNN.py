import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes: int,lr: int, epochs: int, local_epochs: int, bs: int, gamma: int):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.groupnorm1 = nn.GroupNorm(4, 32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(4, 64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.groupnorm3 = nn.GroupNorm(4, 128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.groupnorm4 = nn.GroupNorm(4, 256)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.lr = lr
        self.epochs = epochs
        self.local_epochs = local_epochs
        self.bs = bs
        self.gamma = gamma

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.groupnorm1(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.groupnorm2(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.groupnorm3(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = F.relu(self.groupnorm4(self.conv8(x)))
        x = F.relu(self.conv9(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes: int, lr: float, epochs: int, local_epochs: int, bs: int, gamma: float):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.groupnorm1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(4, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.groupnorm3 = nn.GroupNorm(4, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.lr = lr
        self.epochs = epochs
        self.local_epochs = local_epochs
        self.bs = bs
        self.gamma = gamma

    def forward(self, x):
        x = F.relu(self.groupnorm1(self.conv1(x)))
        x = F.relu(self.groupnorm2(self.conv2(x)))
        x = F.relu(self.groupnorm3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

