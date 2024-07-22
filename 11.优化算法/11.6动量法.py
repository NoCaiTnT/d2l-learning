# 动量法(冲量法)
#   使用平滑过的梯度对权重更新
#       g_t = 1/b * Σ_{i∈I_t} ∇l_i(x_{t-1})
#       v_t = β * v_{t-1} + g_t     w_t = w_{t-1} - η * v_t
#        ↑
#       梯度平滑: v_t = g_t + β * g_{t-1} + β^2 * g_{t-2} + ...
#       β是冲量参数, 常见取值为: 0.5, 0.9, 0.95, 0.99
#
#   更新梯度时, 会考虑之前的梯度, 使得不要变化的太剧烈, 导致过快的更新

import torch
from d2l import torch as d2l

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
d2l.plt.show()

def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()