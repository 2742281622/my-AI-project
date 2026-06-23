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

def softmax_loss(X, t):# Softmax函数的损失函数
    y = softmax(X)
    return cross_entropy_error(y, t)

def L2_loss(Lambda, params):# L2正则化损失函数
    """
    L2正则化损失函数
    params: 包含权重和偏置参数的字典
    Lambda: 正则化系数，设置得越大，对大的权重施加的惩罚就越重
    """
    weight_decay = 0
    for i in range(int(len(params)/2)):
        weight_decay += 0.5 * Lambda * np.sum(params['W'+str(i+1)] ** 2)
    return weight_decay


def L1_loss(Lambda, params):# L1正则化损失函数
    """
    L1正则化损失函数
    params: 包含权重和偏置参数的字典
    Lambda: 正则化系数，设置得越大，对大的权重施加的惩罚就越重
    """
    weight_decay = 0
    for i in range(int(len(params)/2)):
        weight_decay += Lambda * np.sum(np.abs(params['W'+str(i+1)]))
    return weight_decay

def L_infinity_loss(Lambda, params):# L无穷正则化损失函数
    """
    L无穷正则化损失函数
    params: 包含权重和偏置参数的字典
    Lambda: 正则化系数，设置得越大，对大的权重施加的惩罚就越重
    """
    weight_decay = 0
    for i in range(int(len(params)/2)):
        weight_decay += Lambda * np.linalg.norm(params['W'+str(i+1)],np.inf)
    return weight_decay