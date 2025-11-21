---
title: LSTM与GRU
published: 2025-11-20
description: 门机制学习笔记
draft: false
---

# 门控循环单元（GRU）

首先来说GRU,他引入了两个门：**更新门（Reset Gate）**和**重置门（Update Gate）**。其次加入了候选隐藏状态。最后算出最终状态

大致流程就是：输入之后计算两个门的值，然后依次计算隐藏状态的最终状态，最后更新最终状态

## 1、重置门（Reset Gate）

- **作用**：控制前一时刻的隐藏状态有多少信息需要被“遗忘”或“重置”，用于计算当前时刻的候选隐藏状态。
- **工作原理**：重置门的值越小，意味着在计算候选隐藏状态时，越忽略之前的隐藏状态。
- **公式**： $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) $

- $[h_{t-1}, x_t]$表示向量拼接。
- **`σ`** 是Sigmoid函数，将门的值压缩到0和1之间。
- $W_r,b_r$是网络需要学习的参数。

## 2、更新门（Update Gate）

- **作用**：这是一个非常重要的门，它**同时控制着遗忘和记忆**。它决定了有多少旧信息被保留，有多少新信息被加入。
- **工作原理**：类似于LSTM的遗忘门和输入门的结合。
  - $z_t$接近1：倾向于更新。
  - $z_t$接近1：倾向于保留旧状态。
- **公式**：$ z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$
- $[h_{t-1}, x_t]$表示向量拼接。
- **`σ`** 是Sigmoid函数，将门的值压缩到0和1之间。
- $W_z,b_z$ 是网络需要学习的参数。

 ## 3、候选隐藏状态

公式：$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$

- **`⊙`** 表示逐元素相乘 (Hadamard积)。
- **`tanh`** 是激活函数，将输出规范到(-1, 1)之间。
- **`W, b`** 是另一组可学习参数。

## 4、最终状态

公式：$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

- $h_t$: 时间步 t*t* 的**最终隐藏状态**，也是该时间步的输出
- $h_{t-1}$: 前一个时间步的隐藏状态
- $z_t$: **更新门**的输出，范围在 (0,1)(0,1) 之间
- $\tilde{h}_t$: **候选隐藏状态**
- ⊙: **逐元素相乘**（Hadamard积）

------------

# 长短期记忆网络（LSTM）

1. **细胞状态 ($C_t$)**：贯穿时间的“记忆高速公路”，承载长期记忆。
2. **隐藏状态 ($h_t$)**：当前时间步的“输出”或“工作记忆”，包含短期上下文信息。
3. **三个门**（使用Sigmoid函数，输出0到1的值）：
   - **遗忘门 ($f_t$)**：决定从细胞状态中丢弃什么信息。
   - **输入门 ($i_t$)**：决定哪些新信息将存入细胞状态。
   - **输出门 ($o_t$)**：决定基于细胞状态，要输出什么信息。
4. **候选细胞状态 ($\tilde{C}_t$)**：一个包含潜在新信息的备选列表。

## 1、遗忘门（Forget Gate）

**作用**：决定从细胞状态中**丢弃或遗忘**哪些信息。

**公式**：$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$

**工作原理**：

- 查看当前输入 $x_t$和上一个隐藏状态 $h_{t-1}$
- 通过Sigmoid函数为细胞状态 $C_{t-1}$ 中的每个元素输出一个0到1之间的值
- **1** 表示“完全保留这个信息”
- **0** 表示“完全忘记这个信息”

**示例**：在语言模型中，当看到新的主语时，遗忘门可能决定忘记前面句子的旧主语信息。

## 2. 输入门（Input Gate）

**作用**：决定要将**哪些新信息存储到**细胞状态中。

**公式**：$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$

$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$

**工作原理**：

- **输入门层（$i_t$）**：Sigmoid层决定我们要更新哪些值
- **候选值层（$\tilde{C}_t$）**：Tanh层创建新的候选值向量，这些值可能会被加入到细胞状态中

**示例**：当看到新的重要信息时，输入门决定将这些信息加入到长期记忆中。

## 3、更新细胞状态

- **操作**：这是LSTM最核心的一步，结合了前两步的结果来实际更新长期记忆。

- **公式**：

  $ C_t = f_t \odot \ C_{t-1}+i_t \odot \ \tilde{C}_t$

- **目的**：

  - $f_t \odot \ C_{t-1}$：**选择性遗忘**。将旧状态与遗忘门逐元素相乘，丢弃掉我们决定要忘记的信息。
  - $i_t \odot \ \tilde{C}_t$：**选择性记忆**。将输入门与候选状态逐元素相乘，筛选出真正有价值的新信息。
  - **`+`**：将过滤后的旧记忆和筛选过的新信息**相加**，形成更新的长期记忆 `C_t`。这个**加法操作**是解决梯度消失问题的关键。

## 4、输出门（Output Gate）

这个就是最终输出

**作用**：基于当前的细胞状态，决定下一个**隐藏状态**（也就是输出）是什么。

**公式**：$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

$ h_t = o_t \odot \tanh(C_t)$

**工作原理**：

- 首先用Sigmoid层决定细胞状态的哪些部分将被输出
- 然后将细胞状态通过tanh（将值规范到-1到1之间）
- 最后与输出门的输出相乘，得到最终的隐藏状态

# 简单例子

这里以lstm为例

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

# 序列长度（一次输入的字符数）
seq_length=10
# 嵌入向量维度（将字符索引映射为向量）
embed_size=32
# LSTM 隐藏状态维度
hidden_size=64
learning_rate=0.01#学习率
target_size=1#预测个数
epochs=500#学习轮次

inputs=[]
targets=[]
for i in range(len(text)-seq_length):
    seq=text[i:i+seq_length]
    target=text[i+seq_length:i+seq_length+target_size]
    inputs.append([char_to_idx[char] for char in seq])
    targets.append([char_to_idx[char] for char in target])

inputs=torch.tensor(inputs,dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long).squeeze()#因为截取的时候，单次放入的是数组，所以最后是一个二维数组，需要将其展为一维数组 
batch_size = len(inputs)

class Model(torch.nn.Module):
    def __init__(self,vocab_size, embed_size, hidden_size):#词汇长度, 嵌入维度与记忆容量
        super().__init__()
        #嵌入层，将字符索引映射为低维向量，可以理解为将字符转化为一个特征数组
        self.embedding=torch.nn.Embedding(vocab_size, embed_size)
        self.lstm=torch.nn.LSTM(input_size=embed_size,  # 输入维度=嵌入层输出维度
            hidden_size=hidden_size,  # 隐藏层维度
            num_layers=1,  # 单层RNN（简单模型）
            batch_first=True)
        self.linear=torch.nn.Linear(hidden_size,size)#根据特征数目算出每个值的概率

    def forward(self,x):
        embed=self.embedding(x)
        outputs,hidden=self.lstm(embed)#返回一个元组（所有状态，最后一步的隐藏状态）
        output=outputs[:,-1,:]#取最后一次，其实就是hidden
        logits = self.linear(output)#将隐藏状态映射为每个字符的分数 
        return logits
    
model=Model(size, embed_size, hidden_size)
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

test = []


def char_to_idx_func(string):
    idx=[char_to_idx[char] for char in string]
    return idx

# 预处理：将每个字符串转换为字符索引列表
test_idx = []
for t in test:
    if len(t) != seq_length:
        print(f"输入 '{t}' 长度不符，跳过")
        continue
    idx = [char_to_idx[char] for char in t]
    test_idx.append(idx)

# 转换为张量
test=torch.tensor(test_idx, dtype=torch.long)

#测试
with torch.no_grad():
    model.eval()
    while True:
        text=input("请输入测试字符串（输入exit退出）：")
        if text.lower()=='exit':
            break
        test_input=char_to_idx_func(text)
        test_tensor=torch.tensor([test_input], dtype=torch.long)
        test_output=model(test_tensor)
        predicted_idx=torch.argmax(test_output, dim=1).item()
        print(f"预测下一个字符: '{idx_to_char[predicted_idx]}'")
```



