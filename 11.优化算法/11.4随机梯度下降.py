# 随机梯度下降
#   有n个样本时, 计算f(x) = 1/n * Σ_{i=0}^n l_i(x)的导数太贵
#   随机梯度下降在时间t随机选择一个样本t_i, 来近似f(x)
#       x_t = x_{t-1} - η_t * ∇l_{t_i}(x_{t-1})
#       ∇l_{t_i}(x) 的期望 = ∇f(x) 的期望
#   即用一个样本的梯度来近似整体的梯度

import math
import torch
from d2l import torch as d2l

def f(x1, x2):  # 目标函数
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # 目标函数的梯度
    return 2 * x1, 4 * x2

def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 模拟有噪声的梯度
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # 常数学习速度
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
d2l.plt.show()