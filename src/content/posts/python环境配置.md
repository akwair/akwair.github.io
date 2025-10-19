---
title: python环境配置
published: 2025-10-19
description: python环境配置以及anacoda环境配置
draft: true
---

# 一、python下载

1、首先我们需要从官网下载python,（下载地址）[[下载Python |Python.org](https://www.python.org/downloads/)]

2、下载完成后，启动安装程序即可，记得在安装过程中勾选Add python x.x to PATH

3、验证是否安装完成，在cmd中输入一下指令：

```shell
python --version
```

如果返回的是对应的python版本，那么就安装成功了

4、允许脚本的运行：

由于windows的执行策略限制了脚本运行，所以首次运行需要更改执行策略

首先使用管理员方式打开cmd。然后输入一下指令:

```shell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

# 二、vscode下载与配置

1、首先，下载vscode,(下载网址)[https://code.visualstudio.com/]

2、你要注意。vscode是一款编辑器，也就是说，它本身是不能进行编译的，但依托于他强大的

