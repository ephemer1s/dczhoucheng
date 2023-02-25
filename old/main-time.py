import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import csv

# 搭建1D-CNN网络
net = nn.Sequential(
    nn.Conv1d(1, 64, 3, 2, 1),
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
net=net.cuda()
print(net)

# 初始化网络参数
batch_size = 64
learning_rate = 0.02
num_epoches = 20
# 初始化训练参数
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  # 构建随机梯度下降优化器, lr为学习率
loss_func = torch.nn.CrossEntropyLoss()  # 构建交叉熵损失函数
losses = []
acces = []
eval_losses = []  # 测试集损失和准确率
eval_acces = []

# 读入训练集并整理
train_set = np.loadtxt("train_remastered.csv", delimiter=",", dtype=("float32"), skiprows=1)
label = torch.from_numpy(train_set[:, [-1]]).long().cuda()
tseq = np.array(train_set[:, 1:-1])

# 计算数据特征
feature=torch.from_numpy(tseq).cuda()
deal_set = TensorDataset(feature, label)
train_data = DataLoader(dataset=deal_set, batch_size=batch_size, shuffle=True, num_workers=0)

# 循环执行epoch
for e in range(num_epoches):
    train_loss = 0
    train_acc = 0
    net.train()  # 网络开始训练
    for im, label in train_data:
        im = Variable(torch.unsqueeze(im, dim=1).float())
        label = Variable(label)
        # 前向传播
        out = net(im).cuda()
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

    print("epoch %d finished: loss=%6f,  acc=%6f"
          % (e, train_loss / len(train_data), train_acc / len(train_data)))
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

# 绘制测试曲线
fig = plt.figure()
ax = fig.add_subplot(121)
bx = fig.add_subplot(122)
ax.set_title('train loss')
ax.plot(np.arange(len(losses)), losses)
bx.set_title('train acc')
bx.plot(np.arange(len(acces)), acces)

# 读入测试集
test_set = np.loadtxt("test_remastered.csv", delimiter=",", dtype=("float32"), skiprows=1)
tseq = np.array(test_set[:, 1:])
feature=torch.from_numpy(tseq).cuda()
deal_set = TensorDataset(feature)
test_data = DataLoader(dataset=deal_set, batch_size=64, shuffle=False, num_workers=0)

# 对测试集给出预测
net.eval()  # 将模型改为预测模式
res = []
for im in test_data:
    im = Variable(torch.unsqueeze(im[0], dim=1))
    out = net(im).cuda()
    _, pred = out.max(1)
    pred.unsqueeze_(1)
    res.append(pred)
res = torch.cat(res, dim=0)
print(res.size())
res = res.numpy().squeeze()
print(res.shape)
print(res)

# 将结果保存到pred.csv
headers = ['id', 'label']
rows = np.arange(1, 529)
rows = np.vstack((rows, res)).transpose(1, 0)
print(rows)
with open('pred.csv', 'w', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
