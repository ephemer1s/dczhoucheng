import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

train = np.loadtxt("train_remastered.csv",
                   delimiter=",",
                   dtype="float32",
                   skiprows=1)  # 加载训练集
train = train[:, 1:6001]

ff_train = np.array(fft(train[1, :]), dtype=complex)
freq = np.array([np.angle(ff_train), np.abs(ff_train)])
plt.plot(freq[1])
plt.title("LABEL0")
# plt.show()
fig = plt.figure()
x = []
rep = [62, 8, 40, 45, 31, 23, 26, 28, 16]
for i in range(0, 9):
    x.append(fig.add_subplot(331 + i))
    freq = np.abs(np.array(fft(train[rep[i], :]), dtype=complex))
    x[i].plot(freq)
    j = i + 1
    j = "Label" + str(j)
    x[i].set_title(j)
# fig.subplots_adjust(left=0, top=20, right=5, bottom=5, wspace=0.01, hspace=0.01)
fig.show()

fjg = plt.figure()
rep = [0, 4, 6, 14, 26, 27, 30, 34, 43]
x = []
for i in range(0, 9):
    x.append(fjg.add_subplot(331 + i))
    freq = np.abs(np.array(fft(train[rep[i], :]), dtype=complex))
    x[i].plot(freq)
fjg.show()
plt.pause(0)
