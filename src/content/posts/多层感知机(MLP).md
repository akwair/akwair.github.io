---
title: 多层感知机(MLP)
published: 2025-10-23
description: 多层感知机学习笔记
draft: false
---

# 一、什么是神经网络

在说什么是多层感知机之前，我要说说什么是神经网络。

在传统的机器学习流程中，我们对原始数据进行人为特征提取之后，在进行线性模型的预测，最后输出结果，而神经网络就是将人为提取特征变为了通过神经网络提取特征

根据不同的神经网络来抽取特征主要可以分为以下几类：

- 多层感知机
- 卷积神经网络
- 循环神经网络
- Transfromer



# 二、多层感知机(MLP)

MLP由输入层，多个隐藏层和输出层组成,大致公式为：

$$ \mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

其中$$\mathbf{W} \in \mathbb{R}^{m \times n}, \mathbf{b} \in \mathbb{R}^m$$ ,m为输出向量长度，n为输入，最终的y$$\in \mathbb{R}^m$$ 

全连接层：你会发现矩阵的运算长得想把所有的参数连接起来了一样，所以矩阵的变化成为全连接

你会发现这个式子和线性回归十分相似，线性回归的m=1，线性模型都可以表示为输入到全连接层再到输出，所以我们可以把线性模型理解为单层感知机（无隐藏层）。

而对于有些问题，我们会发现，单纯的直线关系是无法解决的，相反，某些曲线的拟合度会更加的好。所以我们会使用非线性模型，采用多个全连接层，但是你会发现，简简单单多个全连接层最终只会得到线性模型，故此我们会在里面加入激活函数（非线性变化）,如：

$$\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}, \text{ReLU}(x) = \max(x, 0)$$

多个激活函数和全连接层就组成了多层感知机（含有多个隐藏层）

所以我们可以通俗直白点来讲，多层感知机就是使用曲线来拟合变化趋势来进行预测或者分类，相当于多个线性模型之间参杂了激活函数

### 核心区别

| 特性       | 单层感知机           | 多层感知机                     |
| ---------- | -------------------- | ------------------------------ |
| 结构       | 无隐藏层             | 至少 1 个隐藏层                |
| 非线性能力 | 仅能处理线性可分问题 | 可处理非线性问题（因激活函数） |
| 表达能力   | 有限（线性模型）     | 强（万能近似定理）             |



# 三、多层感知机预测房价

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import SGD
from sklearn.metrics import r2_score

data=fetch_california_housing()

X,Y=data.data,data.target
#标准化特征值
scaler=StandardScaler()
X=scaler.fit_transform(X)

#随机分配训练集和测试集
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=10)
#转化为张量
Xtrain=torch.tensor(Xtrain,dtype=torch.float32)
Xtest=torch.tensor(Xtest,dtype=torch.float32)
Ytrain=torch.tensor(Ytrain.reshape(-1,1),dtype=torch.float32)
Ytest=torch.tensor(Ytest.reshape(-1,1),dtype=torch.float32)
#搭建模型
class Model(torch.nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_size,4),
            nn.ReLU(),#sigmoid激活效果不好，这里采用relu
            nn.Linear(4,1) 
        )
    #前向传播    
    def forward(self,input):
        return self.layers(input)

#实例化模型，构建损失函数与优化函数
model=Model(Xtrain.shape[1])
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)#传入模型所有参数，设置梯度下降步长

#在训练集上训练模型
for i in range(1000):
    model.train()
    optimizer.zero_grad()
    pre=model(Xtrain)
    loss=criterion(pre,Ytrain) 
    if((i+1)%100==0):
        print(f"loss:{loss.item():.6f}")
    loss.backward()
    optimizer.step()

#测试模型
model.eval()
with torch.no_grad():
    result=model(Xtest)
    R=r2_score(Ytest.numpy(),result.numpy())
    print(f"R平方={R:.6f}")
```

