import numpy as np


class Get_Params:
    def __init__(self, network_shape=[], init_mode='Xavier'):
        '''
        初始化网络参数
        :param network_shape: 网络结构，例如[784, 50, 10]
        :param init_mode: 初始化模式，'Xavier'或'He'
        :return:
        '''
        self.params = {}

        if init_mode == 'Xavier':
            self.params['W1']=np.random.randn(network_shape[0], network_shape[1]) * np.sqrt(1.0 / network_shape[0])
            self.params['b1']=np.zeros(network_shape[1])
            for i in range(1, len(network_shape)-1):
                scale = np.sqrt(1.0 / network_shape[i])
                self.params['W'+str(i+1)]=np.random.randn(network_shape[i], network_shape[i+1]) * scale
                self.params['b'+str(i+1)]=np.zeros(network_shape[i+1])
        elif init_mode == 'He':
            self.params['W1']=np.random.randn(network_shape[0], network_shape[1]) * np.sqrt(2.0 / network_shape[0])
            self.params['b1']=np.zeros(network_shape[1])
            for i in range(1, len(network_shape)-1):
                scale = np.sqrt(2.0 / network_shape[i])
                self.params['W'+str(i+1)]=np.random.randn(network_shape[i], network_shape[i+1]) * scale
                self.params['b'+str(i+1)]=np.zeros(network_shape[i+1])

    def Get(self):
        return self.params