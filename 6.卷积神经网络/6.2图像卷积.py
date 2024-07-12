# 二维交叉相关
#   input * kernel = output
# 二维卷积层
#   输入X: n_h × n_w
#   核W: k_h × k_w
#   偏差b
#   输出Y: (n_h - k_h + 1) × (n_w - k_w + 1) 这里默认步长为1
#       Y = X * W + b
#   W和b是可学习的参数
# 总结
#   卷积层是将输入和核矩阵进行交叉相关, 加上偏移后得到输出
#   核矩阵和偏移是可学习的参数
#   和矩阵的大小是超参数

import torch
from torch import nn
from d2l import torch as d2l
# 图像卷积
# 互相关运算, 步长为1
def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))

# 卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 卷积层的简单应用: 检测图像中不同颜色的边缘
# 定义图像
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
# 定义卷积核
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)
# 该卷积核只能检测竖线, 不能检测横线
print(corr2d(X.t(), K))
# 下面检测横线
print(corr2d(X.t(), torch.tensor([[1.0],
                                  [-1.0]])))

# 学习由X生成Y的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
print(conv2d.weight.data.reshape((1, 2)))
