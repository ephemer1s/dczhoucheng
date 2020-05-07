import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def fastft(a):
    x = fft(a)
    y = np.angle(x)
    z = np.abs(x)
    return y, z


train_set = np.loadtxt("train_remastered.csv", delimiter=",", dtype=("float32"), skiprows=1)
tseq = np.array(train_set[:, 1:-1])
label = np.array(train_set[:, [-1]])

plt.plot(fastft(tseq[1,:]))
plt.show()