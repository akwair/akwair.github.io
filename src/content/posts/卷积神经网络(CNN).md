---
title: 卷积神经网络（CNN）
published: 2025-10-24
description: 卷积神经网络学习笔记
draft: false
---

当你采用全连接层对图片特征进行提取的时候，你会发现输入输出的矩阵、学习的参数会十分庞大，所以采用MLP会十分不理想，所以出现了卷积神经网络

# 一、组成

和MLP相比，卷积神经网络主要分为四层，卷积层、激活层、池化层、全连接层，一般先用卷积层激活层外加池化层进行特征提取，最后使用全连接层进行预测和分类

# 二、卷积层

## 局部感知

相对于MLP的输出是对于所有的参数进行加权，卷积层只输入一个k乘k的小正方形中的数据，然后加权求和，即只做局部运算，这一个k乘k的正方形叫做卷积核（kernel），代表着某一个特征(如绿色通道、红色通道、纹理等)

## 参数共享

同一卷积核在整个图像上滑动时，使用相同的权重参数，既减少了模型参数量，又保证了特征在图像不同位置的一致性（如识别 “猫耳朵” 时，无论在图像左侧还是右侧，判断标准一致）。

# 三、激活层

如MLP中一样，这里的激活层同样是为了使一个线性模型变为非线性模型，可采用同样的如Relu、Sigmoid等激活函数

# 四、池化层

这里我要着重说一下池化层，为什么要有这个东西呢，你会发现，在卷积神经网络中，特征的存在不仅仅依赖像素本身，还依赖于位置关系，比如我现在有一条横线，我的模型可以识别他，但是我把所有的像素打乱之后，我的像素总量是不变的，但是我无法组成横线了。

这个时候，为了防止微小的位置变动影响我们整体的判断，我们引入了池化层

- 最大池化：取窗口内（卷积核）所有元素的最大值作为输出，以保留区域内最显著的特征
- 平均池化：取窗口内所有元素的平均值作为输出，以保留区域内的整体特征

需要注意的是，池化层是没有参数需要学习的，他的操作规则已经被规定了

# 五、全连接层

如前文一样进行线性变化，在cnn中通常位于网络末端进行特征提取和最终预测的

# 六、反向传播与参数优化

在卷积神经网络中，参数的优化依旧可以采取梯度下降的方式，通过反向传播（通过损失来回溯修改参数）

对输出层、全连接层、激活层、池化层、卷积层进行梯度下降算法来迭代参数，详细算法见前文

# 七、卷积神经网络识别CIFAR10

```python
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn import ReLU

transfrom=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])#

train_dataset=datasets.CIFAR10(root="./datasets",
                               train=True,
                               transform=transfrom,
                               download=True)

test_dataset=datasets.CIFAR10(root="./datasets",
                               train=False,
                               transform=transfrom,
                               download=True)

#分批次随即下降，需要加载器来分批次取出来
train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=100,shuffle=False)
print(train_dataset.root)

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),#激活函数，非线性变化
            nn.MaxPool2d(2,2),#最大池化处理边界

            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.layers=nn.Sequential(
            nn.Linear(in_features=64*8*8,out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64*8*8)  # 展平特征图
        x = self.layers(x)
        return x
        
device = torch.device("cpu")
model = model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.02,  # CPU训练时，可适当增大学习率（CPU计算慢，减少收敛时间）
    momentum=0.9,  # 动量项加速收敛
    weight_decay=1e-4  # L2正则化防过拟合
)

model.train()
num_epochs = 20
print("开始训练...")
for epoch in range(num_epochs):
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播+反向传播+参数更新
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 每5轮打印一次训练状态（可选，可删除）
    if (epoch+1) % 2 == 0:
        print(f"第{epoch+1}轮 | 平均损失: {total_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"最终测试准确率: {test_acc:.2f}%")
```

