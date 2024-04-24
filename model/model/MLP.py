import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_classes: int,lr: int, epochs: int, local_epochs: int, bs: int, gamma: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lr = lr
        self.epochs = epochs
        self.local_epochs = local_epochs
        self.bs = bs
        self.gamma = gamma

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
