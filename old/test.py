import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import mycnn

input = torch.randn(10, 64, 6000)
m = nn.Sequential(
    nn.Conv1d(64, 64, 3, 2, 1),
    nn.BatchNorm1d(64),
    nn.ReLU(),

    nn.Conv1d(64, 32, 3, 2, 1),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.MaxPool1d(3, 2),

    nn.Conv1d(32, 16, 4, 3, 2),
    nn.BatchNorm1d(16),
    nn.ReLU(),

    nn.Conv1d(16, 16, 4, 2, 2),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.MaxPool1d(2, 2),

    nn.Conv1d(16, 16, 4, 2, 2),
    nn.BatchNorm1d(16),
    nn.ReLU(),

    nn.Linear(32, 1),
)

print(input.size())
output = m(input)
print(output.size())
