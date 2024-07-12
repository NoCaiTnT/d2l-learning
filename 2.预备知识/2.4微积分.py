import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l

# 导数: f'(x) = (f(x+h)-f(x)) / h 当h趋近于0的时候, 趋近于导数
def f(x):
    return 3 * x ** 2 - 4 * x
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h
h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

x = np.arange(0, 3, 0.1)
d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
d2l.plt.show()

# 导数分子布局: 按分子的行求导, 求导结果的维度 = 分子维度 + 分母反向维度
# 例如: 分子Y的维度为(m,l), 分母X的维度为(n, k), 结果的维度为(m, l, k, n)

# 梯度: 连结一个多元函数对其所有变量的偏导数
# f(x) = Ax, f'(x) = A的转置
# f(x) = (x的转置)A, f'(x) = A
# f(x) = (x的转置)Ax, f'(x) = (A + A的转置)x
# f(x) = ||x||² = (x的转置)x, f'(x) = 2x   注: ||x||为x的范数
