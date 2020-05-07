# 其实是中期汇报文档啊
# 数据分析

> **我的数学很烂。**
>
> ——我自己说的

## 数据集特点

在竞赛官网标明了这个注意事项：

* **注意：同一列的数据不一定是同一个时间点的采样数据，即不要把每一列当作一个特征**

这个说法实在模糊，我们组内产生了两种不同的看法：

* 列数据是对某个轴承的振动在小时间间隔内随机采样得到的
* 列数据是对某个轴承的振动在小时间间隔内均匀采样得到的，但起点时刻不同

针对这两种对信号的理解我们都进行了不同的处理和训练的实验。

**个人偏向于第二种情况。**

#### 如果是随机采样

针对这种可能，我们对余弦函数进行了一个随机抽样的模拟。

![1588870294242](C:\Users\Ephemeris\AppData\Roaming\Typora\typora-user-images\1588870294242.png)

original & different random sample

![1588870307662](C:\Users\Ephemeris\AppData\Roaming\Typora\typora-user-images\1588870307662.png)

Fourier transform

![1588870319978](C:\Users\Ephemeris\AppData\Roaming\Typora\typora-user-images\1588870319978.png)

inverse Fourier transform

随机采样得到的序列，进行傅里叶变换之后当作均匀序列不现实，更不可能随机还原



## 任务特点

#### 模型简单

信号处理的重要性远高于训练网络的重要性。

只需要找到合理表示信号的feature，之后喂给神经网络就能出结果。

完成任务的关键是找到合适的特征量来描述样本。

特征选取不恰当的话基本都会陷入局部最优的情况。

#### 数据简单

信号是一维的，数学苦手也可以处理。不需要高维的傅里叶变换。数据集不大，不用batch也可以跑得很开心

#### 数据集格式比较魔鬼

label和feature粘在了一个文档里，行列均有表头，**给预处理造成了非常大的不必要的麻烦**。

而且数据集前三个字节自带了完全没必要的BOM标识，导致了我读入数据的前两个字节出现了乱码。

~~用excel另存了一遍才修好~~

#### 给出的除label之外的6000个列完全不能当作特征来用

这个我也是试过的，而且是一上来拿到任务就尝试了。

我的想法很简单，能跑不能跑打一杆子就完了。

然后我就写了一个最基本的4层的BP网络：` trainer-classifier.py`

![1588871097698](C:\Users\Ephemeris\AppData\Roaming\Typora\typora-user-images\1588871097698.png)

然后我拿去跑了一下，发现……

![1588871438604](C:\Users\Ephemeris\AppData\Roaming\Typora\typora-user-images\1588871438604.png)

虽然loss看起来是正确的但是这个准确率显然是不如我直接蒙label==0来的划算

显然除了这个网络本身太low以外处理数据也是很关键的因素



# 解决方案

![1588865581650](C:\Users\Ephemeris\AppData\Roaming\Typora\typora-user-images\1588865581650.png)

总之就是分两步，提取特征和训练。

我们首先想到的是信号的各种特征参数。于是我们就飞快地处理了一下原数据：

```python
def featureparam(a):
    """
    处理信号得到特征参数的函数，输入a为二维的ndarray矩阵，输出feature为二维的张量
    不好用，别用这个
    """
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
```

然后带到网络里求，结果依然不是很好看，所以我们的模型截止到今天仍然处于改进中。

竞赛官网是有人发了py版的答案的，但是那个太长了而且用的是keras我不会。有需要的可以参考一下。

# 项目进度

## 进度计划

| 周次   | 主要任务           | 备注 |
| ------ | ------------------ | ---- |
| 6-8周  | 学习原理、配置环境 | -    |
| 7-9周  | 整理思路、提出方案 | -    |
| 9-12周 | 完成编写并提交     | -    |



## 完成情况

小组成员做了一个最基本的分类器作为练习：` tutorial-classifier.py`

* 该分类器可以随机生成两团位于1，3象限附近的点，并通过简单的BP网络判断点属于哪个团簇。

![1588871996460](C:\Users\Ephemeris\AppData\Roaming\Typora\typora-user-images\1588871996460.png)

小组对于不同的feature进行了多种实验。但是目前结果并不理想：

* 实现了使用快速傅里叶变换（FFT）得到信号的频域表示
* 实现了对信号的基本处理，得到了功率、均值、峰峰值等基本参数
* 经过两次大的版本更新，基于PyTorch实现了基本的神经网络，可以进行训练并得到损失和准确率，并输出预测结果。
  * 虽然现在还不太准
* 掌握了各种数据可视化的基本功，从而更清晰的理解本例的数据。

## 未来设想

进一步处理信号分析

* 滤波，可以使用多种不同的滤波器

* 使用其他的变换方法来处理信号，代替FT

* scipy中更多的关于频谱分析的手段

  ![1588872287644](C:\Users\Ephemeris\AppData\Roaming\Typora\typora-user-images\1588872287644.png)

整个更好用一点的网络。

* 计划使用一维CNN

## 成员分工

组长负责程序的主要架构，组员协助完成编程。三人轮流负责每周宣讲的题目，带领大家共同完成。三人分别负责研究信号分析处理的新方法、研究调试和训练网络及各路资料、教程、文档的整理和分发。
