# 转置卷积 上采样
#   卷积不会增大输入的高宽, 通常要么不变, 要么减半
#   转置卷积则可以用来增大输入高宽
#   Y[i:i+h, j:j+w] += X[x,j]·K
# 为什么称之为"转置"
#   对于卷积Y=X ☆ W
#       可以对W构造一个V, 使得卷积等价于矩阵乘法Y' = VX'
#       这里Y', X'是Y,X对应的向量版本
#   转置卷积则等价于Y' = V^T X'
#   如果卷积将输入从(h,w)变成了(h',w')
#       同样超参数的转置卷积则从(h',w')变成(h,w)
# 重新排列输入和核
#   当填充为p, 步幅为s
#       在行和列之间插入s-1行或列
#       将输入填充k-p-1(k是核窗口)
#       将核矩阵上下、左右翻转
#       然后做正常卷积(填充0, 步幅1)
# 形状换算
#   输入高(宽)为n, 核k, 填充p, 步幅s
#   转置卷积: n' = sn+k-2p-s
#       卷积: n' = 向下取整[(n-k+2p+s)/s] -> n >= sn'+k-2p-s
#   如果让高宽成倍增加, 那么k=2p+s
# 转置卷积和反卷积
#   反卷积是卷积的逆运算, 很少用在深度学习中
#   转置卷积是形状上的卷积, 实质还是卷积, 反卷积神经网络指的是用了转置卷积的神经网络
# 总结
#   转置卷积是一种变化了输入和核的卷积, 来得到上采样的目的
#   不等同于数学上的反卷积操作
import torch
from torch import nn
from d2l import torch as d2l


# 转置卷积
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

# 验证转置卷积
X = torch.tensor([[0.0, 1.0],
                  [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0],
                  [2.0, 3.0]])
print(trans_conv(X, K))

# 使用高级API获得相同结果
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print(tconv(X))

# 填充, 步幅, 多通道
# 填充, 在输出上删除上下左右padding行/列
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))
# 步幅
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))
# 多通道
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)

# 与矩阵变换的联系
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
print(Y)

# 将卷积核变为矩阵, 输入3×3拉成一维向量1×9, 输入2×2拉成一维向量1×4, 因此卷积核变为4×9, (1×9)(9×4) = (1×4)
# 变成全连接计算方法
def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
print(W)

# 卷积
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))
# 转置卷积
Z = trans_conv(Y, K)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))