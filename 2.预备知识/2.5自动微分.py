# 计算图: 将代码分解成操作子, 将计算表示成一个无环图, 可以显示构造(Tensorflow/Theano/MXNet)或隐式构造(PyTorch/MXNet)
# 自动求导的两种模式: 正向累积 和 反向累计
# 正向累计: 从x出发, 向y不断求导
# 反向累计(反向传递): 从y出发, 向x不断求导
# 反向传递的内存复杂的为O(n), 即需要存储正向的所有中间结果, 是导致显存过大的原因之一

import torch

# 求y = 2(x的转置)x的导数
x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
print(x.grad)   # 默认值为None
y = 2 * torch.dot(x, x)
print(y)
y.backward()
print(x.grad)
print(x.grad == 4 * x)

# PyTorch会累积梯度, 需要将之前的清零
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 在实际使用过程中, 一般只对标量求导
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)

# 分离计算: 将某些计算移动到记录的计算图之外
# 计算z = u * x的偏导, 而不是z = x * x * x的偏导
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

# Python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)