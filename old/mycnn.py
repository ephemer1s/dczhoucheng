import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.Conv1d(1, 64, 3, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv1d(64, 32, 3, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3, 2)
        )
        self.conv3 = torch.nn.Sequential(
            nn.Conv1d(32, 32, 4, 3, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            nn.Conv1d(32, 16, 4, 2, 2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.conv5 = torch.nn.Sequential(
            nn.Conv1d(16, 16, 4, 2, 2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.mlp = torch.nn.Linear(32,1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.mlp(x)
        return x
