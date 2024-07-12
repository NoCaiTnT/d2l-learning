# 线性模型: y = <w, x> + b, 可以看作是一个单层神经网络
# 衡量预估质量, 损失函数, 例如平方损失: l(y,y') = 0.5 * (y - y')²
# 训练: 损失最小化
# 梯度下降: 向损失梯度下降最快的方向更新权重(不断沿着梯度的反方向更新求解)
# 小批量随机梯度下降: 在样本中随机抽取小批量b个样本来近似全局损失
# 重要的超参数: 批量大小 和 学习率

import math
import time
import numpy as np
import torch
from d2l import torch as d2l

# 矢量化加速 计算两个全为1的10000维向量的和
n = 10000
a = torch.ones([n])
b = torch.ones([n])
# for循环
c = torch.zeros(n)
timer = d2l.Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')
# 向量加法
timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')

# 正态分布(也叫高斯分布): 随机变量具有均值μ和方差σ²
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
# 再次使用numpy进行可视化
x = np.arange(-7, 7, 0.01)
# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
d2l.plt.show()
