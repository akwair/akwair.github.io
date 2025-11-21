---
title: 循环神经网络（RNN）
published: 2025-10-24
description: 循环神经网络学习笔记
draft: false
---

在前面我们提到了多层感知机和卷积神经网络，但是对于自然语言这种序列数据，他们是无法处理的，所以出现了循环神经网络

# 一、RNN 的核心结构：循环单元与记忆传递

RNN 的基本单元是**循环神经元**，其结构可简化为：

```plaintext
输入序列：x₁ → x₂ → x₃ → ... → xₜ（t为时间步）
    ↓      ↓      ↓           ↓
循环单元：[ ] → [ ] → [ ] → ... → [ ]
    ↓      ↓      ↓           ↓
输出序列：h₁ → h₂ → h₃ → ... → hₜ（hₜ为t时刻的隐藏状态，即“记忆”）
```

- **核心逻辑**：在每个时间步 *t*，循环单元接收当前输入$x_t$ 和上一时间步的隐藏状态 $h_{t-1}$（即 “历史记忆”），输出当前隐藏状态 $h_t$：

  $h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$

  其中：

  - $W_{xh}$是输入到隐藏状态的权重(输入的权重),$W_{hh}$是隐藏状态到自身的权重（体现 “记忆传递”），$b_h$ 是偏置；
  - $ h_{t-1} $是上一时刻的隐藏状态
  - *f* 是非线性激活函数（如 tanh、ReLU）。

- **输出层**：若需要预测，可在每个时间步添加输出层，基于 *h**t* 生成预测结果 *y**t*：

	$y_t = g(W_{hy} h_t + b_y)$

	（*g* 是输出激活函数，如 softmax 用于分类）。

# 二、损失函数

自然语言是离散的符号（词或字符），需先转换为数值形式：

- **词表与索引**：将所有可能的词 / 字符构建成 “词表”，每个词对应一个唯一索引（如 “你”→0，“好”→1，“！”→2…）。
- **One-hot 编码**：用长度为词表大小的向量表示每个词，对应索引位置为 1，其余为 0（如 “好”→`[0,1,0,...]`）。
- **词嵌入**：更高效的表示方式，将词映射到低维连续向量（如 “好”→`[0.1, 0.5, -0.3]`），由模型学习得到。

假设 RNN 生成的是一个词序列$\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_T$，真实序列是 $ y_1$,$y_2$,...,$ y_t $（每个$ y_t$是词的索引或 one-hot 向量），损失计算步骤如下：

#### 1. 交叉熵损失（逐词累加）

对每个时间步*t*，计算生成词与真实词的交叉熵，再求和：

- $L = -\sum_{t=1}^{T} \sum_{i=1}^{V} y_{t,i} \cdot \log(\hat{y}_{t,i})$
- *V*是词表大小；
- $y_{t,i}$,是真实词的 one-hot 编码（只有一个位置为 1）；
- $\hat{y}_1$,*i*是 RNN 预测的第*t*步第*i*个词的概率。

# 三、注意点

由前面公式可以看出，当序列过长会导致早期记忆因为梯度爆炸或梯度消失，这是长期依赖问题。下一篇博客会讲一讲门控机制，可以很好解决这个问题

# 四、简单实现

下面是一个十分简单的RNN模型实现对text的预测

```python
import torch
from torch import nn
from torch import optim

text="Hello, RNN! Recurrent Neural Networks are powerful for sequence tasks. Let's build a simple RNN model."
#构建字符字典，即将所有字符映射为索引
chars=sorted(list(set(text)))
#print(chars,'\n')
char_to_idx={ch:id for id,ch in enumerate(chars)}
#print(char_to_idx,'\n')
idx_to_char={id:ch for id,ch in enumerate(chars)}
#print(idx_to_char,'\n')
size=len(chars)

input_size=10#单次输入的单词数
hidden_size=64#隐藏层数目
learning_rate=0.01#学习率
target_size=1#预测个数
epochs=500#学习轮次

inputs=[]
targets=[]
for i in range(len(text)-input_size):
    input=text[i:i+input_size]
    target=text[i+input_size:i+input_size+target_size]
    inputs.append([char_to_idx[char] for char in input])
    targets.append([char_to_idx[char] for char in target])

inputs=torch.tensor(inputs,dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long).squeeze()#因为截取的时候，单次放入的是数组，所以最后是一个二维数组，需要将其展为一维数组 
batch_size = len(inputs)

class Model(torch.nn.Module):
    def __init__(self,vocab_size,hidden_size):#词汇长度与记忆容量
        super().__init__()
        #嵌入层，将字符索引映射为低维向量，可以理解为将字符转化为一个特征数组
        self.embedding=torch.nn.Embedding(vocab_size, hidden_size)#将特征的数目设置为隐藏层数非必须
        self.rnn=torch.nn.RNN(input_size=hidden_size,  # 输入维度=嵌入层输出维度
            hidden_size=hidden_size,  # 隐藏层维度
            num_layers=1,  # 单层RNN（简单模型）
            batch_first=True)
        self.linear=torch.nn.Linear(hidden_size,size)#根据特征数目算出每个值的概率

    def forward(self,x):
        embed=self.embedding(x)
        outputs,hidden=self.rnn(embed)#返回一个元组（所有状态，最后一步的隐藏状态）
        output=outputs[:,-1,:]#取最后一次，其实就是hidden
        logits = self.linear(output) 
        return logits
    
model=Model(size,hidden_size)
criterion = nn.CrossEntropyLoss()#计算交叉熵
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

model.train()
for epoch in range(epochs):
    outputs=model(inputs)#正向传播
    loss=criterion(outputs,targets)#计算损失
    optimizer.zero_grad()#清空梯度
    loss.backward()#反向传播
    optimizer.step()#更新参数

    if((epoch+1)%10==0):
        print(f"第{epoch+1}次循环，损失为{loss}\n")

test = [
    "Hello, RNN", 
    "Recurrent ", 
    "Let's buil"
]

# 预处理：将每个字符串转换为字符索引列表
test_idx = []
for t in test:
    if len(t) != input_size:
        print(f"输入 '{t}' 长度不符，跳过")
        continue
    idx = [char_to_idx[char] for char in t]
    test_idx.append(idx)

# 转换为张量
test=torch.tensor(test_idx, dtype=torch.long)

#测试
with torch.no_grad():
    model.eval()
    outputs=model(test)#储存输出张量
    result=[]
    for i in range(len(outputs)):#输出的是每个字符的概率，选取最大那个然后将其由索引转化为字符
        result.append(idx_to_char[torch.argmax(outputs[i]).item()])
    print(result)
```



