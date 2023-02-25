import os

import numpy as np
import matplotlib.pyplot as plt


def cut(file):
    res1 = []
    res2 = []
    for each in file:
        tmp = str(each).split("loss=")[1]
        res1.append(float(tmp.split(",  acc=")[0]))
        res2.append(float(tmp.split(",  acc=")[1]))
    return np.array(res1), np.array(res2)


if __name__ == '__main__':
    fft1 = open("comparison/fft1.txt", "r", encoding='utf-8')
    fft2 = open("comparison/fft2.txt", "r", encoding='utf-8')
    fft3 = open("comparison/fft3.txt", "r", encoding='utf-8')

    ifft1 = open("comparison/fpool1.txt", "r", encoding='utf-8')
    ifft2 = open("comparison/fpool2.txt", "r", encoding='utf-8')
    ifft3 = open("comparison/fpool3.txt", "r", encoding='utf-8')

    f1l, f1a = cut(fft1.readlines())
    f2l, f2a = cut(fft2.readlines())
    f3l, f3a = cut(fft3.readlines())

    i1l, i1a = cut(ifft1.readlines())
    i2l, i2a = cut(ifft2.readlines())
    i3l, i3a = cut(ifft3.readlines())

    fl = (f1l+f2l+f3l)/3
    fa = (f1a+f2a+f3a)/3
    il = (i1l+i2l+i3l)/3
    ia = (i1a+i2a+i3a)/3

    x = np.arange(20)
    fig = plt.figure(figsize=(6.4,8), dpi=200)
    ax = fig.add_subplot(211)
    bx = fig.add_subplot(212)
    ax.plot(x, fa, color='red', label='Halved Pooling Layer')
    ax.plot(x, ia, color='blue', label='Full Pooling Layer')
    ax.legend()
    ax.set_title("average acc")
    ax.set_xticks([0,4,8,12,16,20])
    bx.plot(x, fl, color='red', label='Halved Pooling Layer')
    bx.plot(x, il, color='blue', label='Full Pooling Layer')
    bx.legend()
    bx.set_xticks([0,4,8,12,16,20])
    bx.set_title("average loss")
    fig.subplots_adjust(wspace=0, hspace=0.2)
    fig.savefig("2")
