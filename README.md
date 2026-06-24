# my-AI-project
    上传了我进行的一些ai项目,tool为本人纯手搓，使用tool可训练或使用模型，未使用其他框架，暂时只能在cpu运行，暂时仅支持全连接网络训练和使用，后面会不断修改…………

# 目录结构
    project
        -[tool]             神经网络开发和运行的工具包
        -[mnist-FCNN]       手写体数字识别，模型精度91%
            -model.op
            -main.py

# 日志
260624：增加了l1、l2、l∞正则化，新增BCEWithLogitsLoss、MSELoss层以支持二分类、回归任务