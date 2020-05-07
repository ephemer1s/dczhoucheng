import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def featureparam(a):
    x_avg = np.mean(a, axis=1, keepdims=True)  # 算术均值
    x_std = np.std(a, axis=1, ddof=1, keepdims=True)  # 标准差
    x_var = np.var(a, axis=1, ddof=1, keepdims=True)  # 方差
    x_ptp = np.ptp(a, axis=1, keepdims=True)  # 峰峰值
    x_rms = np.sqrt(np.mean(a ** 2, axis=1, keepdims=True))  # 有效值
    x_skw = stats.skew(a, axis=1).reshape(a.shape[0], 1)  # 偏度
    x_kur = stats.kurtosis(a, axis=1).reshape(a.shape[0], 1)  # 峰度
    feature = torch.from_numpy(np.array([x_avg, x_std, x_var,
                                         x_ptp, x_rms, x_skw, x_kur]).squeeze().T)
    return feature


def fastft(a):
    x = np.array([fft(a[i, :]) for i in range(a.shape[0])])
    y = np.array([np.abs(x[i, :]) for i in range(a.shape[0])])
    y = torch.from_numpy(y / (y.max() - y.min()))
    z = torch.from_numpy(np.array([np.angle(x[i, :]) for i in range(a.shape[0])]))

    return y, z


# 搭建BP网络
net1 = nn.Sequential(
    nn.Linear(7, 50),
    nn.ReLU(),
    nn.Linear(50, 200),
    nn.ReLU(),
    nn.Linear(200, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
net2 = nn.Sequential(
    nn.Linear(6000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 200),
    nn.ReLU(),
    nn.Linear(200, 40),
    nn.ReLU(),
    nn.Linear(40, 10)
)

# 初始化网络参数
optimizer = torch.optim.SGD(net2.parameters(), lr=0.03)  # 构建随机梯度下降优化器, lr为学习率
loss_func = torch.nn.CrossEntropyLoss()  # 构建交叉熵损失函数
losses = []
acces = []
eval_losses = []  # 测试集损失和准确率
eval_acces = []

# 读入训练集并整理
train_set = np.loadtxt("train_remastered.csv", delimiter=",", dtype=("float32"), skiprows=1)
label = torch.from_numpy(train_set[:, [-1]]).long()
tseq = np.array(train_set[:, 1:-1])

# 计算数据特征
# feature, _ = fastft(tseq)
# feature = featureparam(tseq)
feature = torch.from_numpy(tseq)
print(feature.size())
deal_set = TensorDataset(feature, label)
train_data = DataLoader(dataset=deal_set, batch_size=64, shuffle=True, num_workers=0)

# 循环执行epoch
for e in range(50):
    train_loss = 0
    train_acc = 0
    net2.train()  # 网络开始训练
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net2(im)
        loss = loss_func(out, label.squeeze())
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
plt.show()
plt.pause(0)
exit(0)

# 读入测试集
test_set = np.loadtxt("test_remastered.csv", delimiter=",", dtype=("float32"), skiprows=1)
tseq = np.array(test_set[:, 1:])
feature = featureparam(tseq)
deal_set = TensorDataset(feature)
test_data = DataLoader(dataset=deal_set, batch_size=32, shuffle=False, num_workers=0)
net1.eval()  # 将模型改为预测模式
res = []
for im in test_data:
    im = Variable(im[0])
    out = net1(im)
    _, pred = out.max(1)
    res.append(pred)
print(res)
