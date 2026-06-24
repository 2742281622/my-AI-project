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
                 epochs=10, batch_size=10, learning_rate=0.01,task='multi-class',regularization='none',
                 lambda_w=0.01,
                 monitor=True, auto_save=False,model_name='model'):
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
        :param task: 任务类型 multi-class 多分类 binary-class 二分类 regression 回归
        :param regularization: 正则化 none 无正则化 l2 l2正则化 l1 l1正则化 L_infinity 无穷范数正则化
        :param lambda_w: 权重正则化参数
        :param monitor: 是否监控性能
        :param auto_save: 是否自动保存模型参数
        :return:
        '''
        self.monitor_flg = monitor# 是否监控性能
        if self.monitor_flg:
            self.monitor = Performance_Monitor()# 性能监控
            self.monitor.start()# 开始监控
        
        self.network={}# 网络结构，用于打印
        self.network['task']=task# 任务类型
        self.network['regularization']=regularization# 正规则化
        self.regularization=regularization# 正规则化
        self.lambda_w=lambda_w# 权重正则化参数
        self.task=task# 任务类型
        self.model_name=model_name# 模型名称
        self.network['Optimizer']=optimizer.__name__
        self.optimizer = optimizer()
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
        self.cache = {}# 记录层参数
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

                self.__Computational(dout=1,backward_flg=True)# 反向计算

                self.__get_grads()# 获取梯度
                for key in self.params.keys():# 对需要的层逐层更新参数

                    if self.regularization != 'none':# 如果有正则化
                        if self.regularization == 'l1':
                            # L1正则梯度：dW += λ * sign(W)
                            self.grads[key]['W'] += self.lambda_w * np.sign(self.params[key]['W'])
                        elif self.regularization == 'l2':
                            # L2权重衰减梯度：dW += λ * W
                            self.grads[key]['W'] += self.lambda_w * self.params[key]['W']
                        elif self.regularization == 'L_infinity':
                            # L∞仅作示例，实际几乎不用
                            self.grads[key]['W'] = self.lambda_w * np.max(np.abs(self.params[key]['W']))
                    
                    self.optimizer.update(self.params[key], self.grads[key])# 更新参数和梯度
            

            test_acc=self.__acc_test()# 测试集accuracy率
            train_acc=self.__acc_train()# 训练集accuracy率
            print('test accuracy:%.2f%%\ttrain accuracy:%.2f%%\tloss:%.12f'% (test_acc,train_acc,self.loss))# 每个epoch打印loss和accuracy率
            if self.monitor_flg:# 如果监控性能
                self.monitor.probe()# 探针
        
    def predict(self, x):# 预测
        self.__Computational(x,train_flg=False)
        if self.monitor_flg:# 如果监控性能
                self.monitor.probe()# 探针
        return self.cache['layer'+str(self.layer_nub-2)]

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
                    self.cache['layer'+str(i+1)]=self.layers['layer'+str(i+1)].forward(x,train_flg=train_flg)
                elif i != self.layer_nub-2:# 隐藏层
                    self.cache['layer'+str(i+1)]=self.layers['layer'+str(i+1)].forward(self.cache['layer'+str(i)],train_flg=train_flg)
                else:
                    if train_flg:# 最后一层 计练
                        self.loss=self.layers['layer'+str(i+1)].forward(self.cache['layer'+str(i)], t)

    def __acc_test(self):# 测试集accuracy率
        batch=self.x_test.shape[0]
        acc=0.00
        try:
            result=self.predict(self.x_test)
        except MemoryError:
            batch=batch//2
            for i in range(0,self.x_test.shape[0],batch):
                result=self.predict(self.x_test[i:i+batch])

        if self.task=='multi-class':# 多分类
            result=np.argmax(result,axis=1)
            t_test=np.argmax(self.t_test,axis=1)
            acc=np.mean(result==t_test)*100
        elif self.task=='binary-class':# 二分类
            result=np.argmax(result,axis=1)
            t_test=np.argmax(self.t_test,axis=1)
            acc=np.mean(result==t_test)*100
        elif self.task=='regression':# 回归
            result=result.reshape(-1)
            t_test=self.t_test
            acc=np.mean((result - t_test) ** 2)
        
        return acc

    def __acc_train(self):# 训练集accuracy率
        batch=self.t_train.shape[0]
        acc=0.00
        try:
            result=self.predict(self.x_train)
        except MemoryError:
            batch=batch//2
            for i in range(0,self.x_train.shape[0],batch):
                result=self.predict(self.x_train[i:i+batch])

        if self.task=='multi-class':# 多分类
            result=np.argmax(result,axis=1)
            t_test=np.argmax(self.t_train,axis=1)
            acc=np.mean(result==t_test)*100
        elif self.task=='binary-class':# 二分类
            result=np.argmax(result,axis=1)
            t_test=np.argmax(self.t_train,axis=1)
            acc=np.mean(result==t_test)*100
        elif self.task=='regression':# 回归
            result=result.reshape(-1)
            t_test=self.t_train
            acc=np.mean((result - t_test) ** 2)
        
        return acc

    def __save_model(self):# 保存模型
        model = {}
        model['network'] = self.network
        model['layers'] = self.layers
        model['params'] = self.params
        with open(self.model_name+'.op', 'wb') as f:
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

    load_models = False

    save_model = False
    model_name = 'model FC00'


    trainer = Trainer(x_train, t_train, x_test, t_test, optimizer=Adam, epochs=1000, batch_size=100,\
                      monitor=True, auto_save=save_model,regularization='none',model_name=model_name)
    


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
        trainer.add_layer(SoftmaxWithLoss)


    trainer.show_network()# 显示网络信息
    trainer.train()
