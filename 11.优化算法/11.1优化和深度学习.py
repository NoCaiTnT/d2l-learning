# 优化问题
#   一般形式: minimize f(x)  subject to x ∈ C
#       目标函数 f: R^n -> R
#       限制集合例子: C = {x| h_1(x)=0,...,h_m(x)=0, g_1(x)≤0,...,g_r(x)≤0}
#       如果C = R^n, 那就是不受限

# 风险 vs 经验风险
#   经验风险是训练数据集的平均损失
#   风险则是整个数据群的预期损失

# 局部最小 vs 全局最小
#   全局最小x*: f(x*) ≤ f(x) 任意 x ∈ C
#   局部最小x*: 存在ε, 使得 f(x*) ≤ f(x) 任意 x: ||x-x*|| ≤ ε
#   使用迭代优化算法来求解, 一般只能保证找到局部最小值
#   因为当到局部最小值时, 梯度为0, 无法继续优化

# 鞍点
#   鞍点是梯度为0, 但既不是局部最小值, 又不是全局最小值
#   同样导致梯度消失


import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l

# 定义了两个函数: 风险函数f, 经验风险函数g
def f(x):
    return x * torch.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)

# 假设我们只有有限的训练数据
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = torch.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
d2l.plt.show()

# 局部/全局最小值
x = torch.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
d2l.plt.show()

# 鞍点
x = torch.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
d2l.plt.show()

# 高纬度鞍点
x, y = torch.meshgrid(
    torch.linspace(-1.0, 1.0, 101), torch.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
d2l.plt.show()

# 梯度消失
# 假设我们想最小化函数f(x)=tanh(x)
# 然后我们恰好从x=4开始
# 正如我们所看到的那样，f的梯度接近零
# 更具体地说, f'(x) = 1-tanh²(x), 因此f'(4)=0.0013
# 因此, 在我们取得进展之前, 优化将会停滞很长一段时间
x = torch.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [torch.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
d2l.plt.show()