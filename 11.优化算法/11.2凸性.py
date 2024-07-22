# 凸集
#   一个R^n的子集C是凸集, 当且仅当
#       αx + (1-α)y ∈ C, 任意x, y ∈ C, 0 ≤ α ≤ 1
#       集合中任意两点的连线, 线上的点都在集合内
#   两个凸集的交集是凸集
#   两个凸集的并集不一定是凸集

# 凸函数
#   函数f: C -> R是凸函数, 当且仅当
#       f(αx + (1-α)y) ≤ αf(x) + (1-α)f(y), 任意x, y ∈ C, 0 ≤ α ≤ 1
#       函数任意两点的连线, 线上的点都在函数图像上方
#   如果x ≠ y, α ∈ (0, 1)时不等式严格成立, 那么叫做严格凸函数

# 凸函数优化
#   如果代价函数f是凸的, 且限制集合C是凸的, 那么优化问题是凸优化问题, 那么局部最小值就是全局最小值
#   严格凸优化问题有唯一的全局最小

# 凸和非凸的例子
#   凸:
#       线性回归f(x) = ||Wx-b||^2
#       Softmax回归
#   非凸:
#       MLP, CNN, RNN, Attention, ...

import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l

# 凸函数
f = lambda x: 0.5 * x**2  # 凸函数
g = lambda x: torch.cos(np.pi * x)  # 非凸函数
h = lambda x: torch.exp(0.5 * x)  # 凸函数

x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
d2l.plt.show()