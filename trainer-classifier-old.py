import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.out = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.out(x)
#         return x


# net = Net(n_feature=6000, n_hidden=30, n_output=10)
net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
loss_count = []
losses = []
acces = []
eval_losses = []
eval_acces = []

train_set = np.loadtxt("train_remastered.csv", delimiter=",", dtype=("float32"), skiprows=1)
feature = torch.from_numpy(train_set[:, 1:-1])
label = torch.from_numpy(train_set[:, [-1]])
# train_ft = np.array([np.abs(fft(train_set[i, 1:6001])) for i in range(tr_feature.shape[0])])
# train_ft = torch.from_numpy(train_ft)
# test_set = np.loadtxt("test_remastered.csv", delimiter=",", dtype=("float32"), skiprows=1)  # import data
# test_set = torch.from_numpy(test_set[:, 1:6001])
# test_data = DataLoader(test_set, batch_size=128, shuffle=False)

deal_set = TensorDataset(feature,label)
train_data = DataLoader(dataset=deal_set, batch_size=64, shuffle=True, num_workers=2)

for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()  # 网络开始训练
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        loss = loss_func(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)  # 挑选出输出时值最大的位置
        num_correct = (pred == label).sum().item()  # 记录正确的个数
        acc = num_correct / im.shape[0]  # 计算精确率
        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
plt.plot(np.arange(len(acces)), acces)
plt.title('train acc')
plt.plot(np.arange(len(eval_losses)), eval_losses)

# epoch = 100
# for t in range(epoch):
#     # out = net(tr_feature)
#     out = net(train_ft)
#     loss = loss_func(out, tr_label)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if t % 10 == 0:
#         loss_count.append(loss)

# plt.figure('Pytorch_CNN_loss')
# plt.plot(loss_count, label='loss')
# plt.legend()
# plt.show()


