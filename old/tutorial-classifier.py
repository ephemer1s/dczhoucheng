from tutorial_net import Net
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fake testing numbers from two clusters "0" and "1"
n_data = torch.ones(100, 2)
s0 = torch.normal(2 * n_data, 1)
l0 = torch.zeros(100)
s1 = torch.normal(-2 * n_data, 1)
l1 = torch.ones(100)

# combining data
s = torch.cat((s0, s1), 0).type(torch.FloatTensor)
l = torch.cat((l0, l1), 0).type(torch.LongTensor)
print(s.shape, l.shape)

net = Net(n_feature=2, n_hidden=10, n_output=2)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
lossfunc = torch.nn.CrossEntropyLoss()

plt.ion()
for t in range(100):
    out = net(s)
    loss = lossfunc(out, l)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = l.data.numpy()
        plt.scatter(s.data.numpy()[:, 0], s.data.numpy()[:, 1],
                    c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x