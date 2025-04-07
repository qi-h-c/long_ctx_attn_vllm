# C++/CUDA Extensions in PyTorch

本文编写了一个 `extension_cpp.ops.add` 自定义运算符，该运算符包括 CPU 和 CUDA 内核的实现。

本示例适用于 PyTorch 2.4+ 版本。

## 功能描述

我们创建了一个名为 `add` 的自定义运算符，它执行两个 tensor 的逐元素加法操作 `a + b`，并且支持 CPU 和 CUDA（GPU）计算。

## 构建方法

To build:
```
pip install --no-build-isolation -e .
```

To test:
```
python test/test_extension.py
```

## 贡献者

- **Liu Jiahui**

## 引用

该项目引用了 [PyTorch C++/CUDA 扩展示例](https://github.com/pytorch/extension-cpp) 项目，用于学习和实现自定义的 C++/CUDA 扩展。