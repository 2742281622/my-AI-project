# coding: utf-8
# 性能监控 用于监视网络训练过程中的性能指标 内存使用、运行时间、loss等

import time
import psutil
import os as os_module

class Performance_Monitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.process = None# 进程对象
        self.memory = 0# 最大内存占用（MB）
    
    def start(self):# 开始监控
        self.start_time = time.time()
        self.process = psutil.Process(os_module.getpid())
    
    def probe(self):# 探针
        # 最大内存占用（MB）
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # 转换为MB
        if current_memory > self.memory:
            self.memory = current_memory

    def end(self):# 结束监控
        self.end_time = time.time()

    def show(self):# 显示监控结果
        using_time=self.end_time - self.start_time
        if using_time > 3600:
            using_time = using_time / 3600
            print(f"运行时间: {using_time:.2f} 小时")
        elif using_time > 60:
            using_time = using_time / 60
            print(f"运行时间: {using_time:.2f} 分钟")
        else:
            print(f"运行时间: {using_time:.4f} 秒")
        
        print(f"最大内存占用: {self.memory:.2f} MB")