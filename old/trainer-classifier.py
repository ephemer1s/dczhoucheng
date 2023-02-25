import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

net = nn.Sequential(
    nn.Linear(6000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 200),
    nn.ReLU(),
    nn.Linear(200, 40),
    nn.ReLU(),
    nn.Linear(40, 10)
)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
loss_count = []
losses = []
acces = []
# eval_losses = []
# eval_acces = []

train_set = np.loadtxt("train_remastered.csv", delimiter=",", dtype=("float32"), skiprows=1)
feature = torch.from_numpy(train_set[:, 1:-1])
label = torch.from_numpy(train_set[:, [-1]]).long()
deal_set = TensorDataset(feature, label)
train_data = DataLoader(dataset=deal_set, batch_size=64, shuffle=True, num_workers=0)

for e in range(100):
    train_loss = 0
    train_acc = 0
    net.train()  # 网络开始训练
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
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

    print("epoch %d: loss=%3f,  acc=%3f"
          % (e, train_loss / len(train_data), train_acc / len(train_data)))
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

fig = plt.figure()
ax = fig.add_subplot(121)
bx = fig.add_subplot(122)
ax.set_title('train loss')
ax.plot(np.arange(len(losses)), losses)

bx.set_title('train acc')
bx.plot(np.arange(len(acces)), acces)
# bx.plot(np.arange(len(eval_losses)), eval_losses)
plt.show()
