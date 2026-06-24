import numpy as np
from tool.Activation_Function import *

def mean_squared_error(y, t):# 均方误差 回归任务
    # 计算均方误差 E=1/2 * sum((y-t)^2)
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):# 交叉熵误差 分类任务
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def binary_cross_entropy(y, t):# 二分类/多标签损失函数
    # BCE 二分类/多标签损失，y是sigmoid输出(0~1)，t是0/1标签
    # L = -sum(t * log(y) + (1 - t) * log(1 - y))
    # 加极小值防止log(0)出现负无穷
    delta = 1e-7
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log(1 - y + delta))

def binary_cross_entropy_batch(y, t):
    delta = 1e-7
    loss = -np.sum(t * np.log(y + delta) + (1 - t) * np.log(1 - y + delta))
    return loss / y.shape[0]


def softmax_loss(X, t):# Softmax函数的损失函数
    y = softmax(X)
    return cross_entropy_error(y, t)
