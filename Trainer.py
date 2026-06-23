# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

from tool.Layers import *# 层
from tool.Optimizer import *# 优化器
from tool.Performance_Monitor import *  # 性能监控
from tool.Activation_Function import *# 激活函数
from tool.Loss_Function import *# 损失函数

from collections import OrderedDict
import pickle # 用于保存模型参数
import atexit # 用于注册退出时的自动保存
import numpy as np

from tool.dataset.mnist import load_mnist


class Trainer:
    def __init__(self, x_train=None, t_train=None, x_test=None, t_test=None, optimizer=Momentum,\
                 epochs=10, batch_size=10, learning_rate=0.01,loss_func=cross_entropy_error,
                 monitor=True, auto_save=False):
        '''
        训练器
        :param x_train: 训练数据
        :param t_train: 训练标签
        :param x_test: 测试数据
        :param t_test: 测试标签
        :param optimizer: 优化器
        :param epochs: 训练轮数
        :param batch_size: 单次训练数据批次大小
        :param learning_rate: 学习率
        :param loss_func: 损失函数
        :param monitor: 是否监控性能
        :param auto_save: 是否自动保存模型参数
        :return:
        '''
        self.monitor_flg = monitor# 是否监控性能
        if self.monitor_flg:
            self.monitor = Performance_Monitor()# 性能监控
            self.monitor.start()# 开始监控
        
        self.network={}# 网络结构，用于打印
        self.network['Optimizer']=optimizer.__name__
        self.optimizer = optimizer()
        self.loss_func=loss_func# 损失函数
        self.network['layers']={}
        self.layers = OrderedDict()# 层结构，用于实例运算
        self.layer_nub=1

        self.epochs = epochs# 训练轮数
        self.batch_size = batch_size# 单次训练数据批次大小
        self.learning_rate = learning_rate# 学习率
        self.loss=None


        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.params = {}# 记录层参数
        self.ache = {}# 记录层参数
        self.dout = {}# 记录x的梯度
        self.grads = {}# 记录层梯度

        if auto_save:# 是否自动保存模型参数
        # 注册退出时的自动保存
            atexit.register(self.__save_model)

    def add_layer(self, layer):# 添加层
        try:# 尝试实例化层
            self.layers['layer'+str(self.layer_nub)]=layer()
        except TypeError:
            self.layers['layer'+str(self.layer_nub)] = layer
        
        self.network['layer_nub']=self.layer_nub# 记录层数
        if type(self.layers['layer'+str(self.layer_nub)])!=type(object):# 如果不是类
            self.network['layers']['layer'+str(self.layer_nub)]=str(self.layers['layer'+str(self.layer_nub)]).split('.')[-1].split(' ')[0]
        else:# 如果是类
            self.network['layers']['layer'+str(self.layer_nub)]=str(self.layers).split('.')[-1].split("'")[0]# 记录层名str

        self.layer_nub+=1
        return self

    def set_layer(self, **params):# 设置层参数
        last_layer_name = 'layer' + str(self.layer_nub - 1)
        layer_class = self.layers[last_layer_name]
        self.layers[last_layer_name] = layer_class(**params)  # 用参数实例化层
        self.params[last_layer_name] = params# 记录层参数

    def train(self):# 训练
        for i in range(self.epochs):
            for j in range(0,self.x_train.shape[0],self.batch_size):# 遍历所有数据
                self.__Computational(self.x_train[j:j+self.batch_size],self.t_train[j:j+self.batch_size])# 前向计算

                if self.network['layers']['layer'+str(self.layer_nub-1)]=='SoftmaxWithLoss':# 如果是SoftmaxWithLoss层
                    pass
                else:
                    self.loss=self.loss_func(self.ache['layer'+str(self.layer_nub-1)],self.t_train[j:j+self.batch_size])# 计算loss
                

                douts=(self.ache['layer'+str(self.layer_nub-2)]-self.t_train[j:j+self.batch_size])/self.batch_size# 计算训练数据的差值

                self.__Computational(dout=douts,backward_flg=True)# 反向计算

                self.__get_grads()# 获取梯度
                for key in self.params.keys():# 对需要的层逐层更新参数
                    self.optimizer.update(self.params[key], self.grads[key])# 更新参数和梯度

            acc=self.__test()# 测试

            print('accuracy:%.2f%%\tloss:%.12f'% (acc,self.loss))# 每个epoch打印loss和accuracy率
            if self.monitor_flg:# 如果监控性能
                self.monitor.probe()# 探针
        
    def predict(self, x):# 预测
        self.__Computational(x,train_flg=False)
        if self.monitor_flg:# 如果监控性能
                self.monitor.probe()# 探针
        return self.ache['layer'+str(self.layer_nub-2)]

    def show_network(self):# 打印网络结构
        print('\n==========================================================')
        print("==\t%s\t=="%('Network information').center(44))
        print('==------------------------------------------------------==')
        
        for key in self.network.keys():
            if isinstance(self.network[key], (int, float, str, bool)) or self.network[key] is None:
                print('==\t%s\t|\t%s\t\t=='%(key.center(11),str(self.network[key]).center(15)))
                print('==------------------------------------------------------==')
            else:
                try:
                    print('==    %s\t|\t\t\t\t=='%(key.center(11)))
                    for x in self.network[key]:
                        print('==\t%s\t|\t%s\t\t=='%(('--'+x).center(13),str(self.network[key][x]).center(15)))
                except TypeError:# 如果是列表或元组
                    print('==\t%s\t|\t%s\t\t=='%(('--'+key).center(15),str(type(self.network[key])).center(15)))

                print('==------------------------------------------------------==')


        print('==========================================================\n')

    def load_model(self, path):# 加载模型
        with open(path, 'rb') as f:
            model_params = pickle.load(f)
        self.network = model_params['network']
        self.layers = model_params['layers']
        self.params = model_params['params']
        self.layer_nub = int(self.network['layer_nub'])+1

    def __Computational(self,x=None,t=None,dout=1,train_flg=True,backward_flg=False):# 逐层计算网络输出
        if backward_flg:# 反向传播
            for i in range(self.layer_nub-1,0,-1):
                if i == self.layer_nub-1:# 最后一层
                    self.dout['layer'+str(i)]=self.layers['layer'+str(i)].backward(dout)
                else:
                    self.dout['layer'+str(i)]=self.layers['layer'+str(i)].backward(self.dout['layer'+str(i+1)])
        else:# 前向传播
            for i in range(self.layer_nub-1):
                if i == 0:# 输入层
                    self.ache['layer'+str(i+1)]=self.layers['layer'+str(i+1)].forward(x)
                elif i != self.layer_nub-2:# 隐藏层
                    self.ache['layer'+str(i+1)]=self.layers['layer'+str(i+1)].forward(self.ache['layer'+str(i)])
                else:
                    if type(self.layers['layer'+str(i+1)])==SoftmaxWithLoss and train_flg:# 最后一层层为SoftmaxWithLoss层
                        self.loss=self.layers['layer'+str(i+1)].forward(self.ache['layer'+str(i)], t)
                    elif train_flg:
                        self.ache['layer'+str(i+1)]=self.layers['layer'+str(i+1)].forward(self.ache['layer'+str(i)], t)

    def __test(self):# 测试
        batch=self.x_test.shape[0]
        try:
            result=self.predict(self.x_test)
        except MemoryError:
            batch=batch/2
            for i in range(0,self.x_test.shape[0],batch):
                result=self.predict(self.x_test[i:i+batch])

        result=np.argmax(result,axis=1)
        t_test=np.argmax(self.t_test,axis=1)
        return (np.mean(result==t_test)*100)

    def __save_model(self):# 保存模型
        model = {}
        model['network'] = self.network
        model['layers'] = self.layers
        model['params'] = self.params
        with open('model.op', 'wb') as f:
            pickle.dump(model, f)

    def __get_grads(self):# 获取层梯度
        try:
            for layer in self.params.keys():
                p_dict=self.layers[layer].__dict__
                if layer not in self.grads:# 如果层梯度不存在，初始化
                    self.grads[layer] = {}
                for name in self.params[layer].keys():
                    self.grads[layer][name]=p_dict['d'+name]
        except:
            pass

    def __del__(self):# 结束时显示监控结果
        if self.monitor_flg:# 如果监控性能
            self.monitor.end()# 结束监控
            self.monitor.show()# 显示监控结果

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)# 数据已打乱
    x_train = x_train[:15000]
    t_train = t_train[:15000]
    x_test = x_test[:5000]
    t_test = t_test[:5000]

    trainer = Trainer(x_train, t_train, x_test, t_test, optimizer=Adam, epochs=100, batch_size=100,\
                      monitor=True, auto_save=False)
    
    load_models = True


    if load_models:
        trainer.load_model('model.op')
    else:
        trainer.add_layer(Affine).set_layer(W=np.random.randn(784, 64), b=np.random.randn(64))
        trainer.add_layer(ReLU)
        trainer.add_layer(Affine).set_layer(W=np.random.randn(64, 64), b=np.random.randn(64))
        trainer.add_layer(ReLU)
        trainer.add_layer(Affine).set_layer(W=np.random.randn(64, 128), b=np.random.randn(128))
        trainer.add_layer(ReLU)
        trainer.add_layer(Affine).set_layer(W=np.random.randn(128, 10), b=np.random.randn(10))
        #trainer.add_layer(SoftmaxWithLoss)
        trainer.add_layer(Softmax)


    trainer.show_network()# 显示网络信息
    trainer.train()
