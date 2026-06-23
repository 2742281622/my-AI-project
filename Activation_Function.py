# coding: utf-8
import numpy as np


def identity_function(x):# 恒等函数
    return x


def step_function(x):# 阶跃函数
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):# Sigmoid函数
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):# Sigmoid函数的导数
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):# ReLU函数
    return np.maximum(0, x)


def relu_grad(x):# ReLU函数的导数
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):# Softmax函数
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

