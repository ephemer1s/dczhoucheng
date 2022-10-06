import torch as t
import torch.nn.functional as F


class Net(t.nn.Module):
    """
    test net for tutorial-classifier.py
    """
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = t.nn.Linear(n_feature, n_hidden)
        self.out = t.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x
