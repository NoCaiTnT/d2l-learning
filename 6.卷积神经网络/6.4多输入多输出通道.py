# 多个输入通道
#   彩色图像可能有RGB三个通道
#   转换为灰度会丢失信息
#   每个通道都有一个卷积核, 结果是所有通道卷积结果的和
#   输入X: c_i × n_h × n_w
#   核W: c_i × k_h × k_w
#   输出Y: m_h × m_w
# 多个输出通道
#   无论有多少输入通道, 到目前为止我们只用到单输出通道
#   可以有多个三维卷积核, 每个核生成一个输出通道
#   输入X: c_i × n_h × n_w
#   核W: c_o × c_i × k_h × k_w           # c_i表示输入图像的维度, 有几维就有几个卷积核, 他们构成1组特征提取器, c_o表示特征提取器的个数, 即对输入图像提取多少个特征
#   输出Y: c_o × m_h × m_w
# 多个输入和输出通道
#   每个输出通道可以识别特定模式
#   输入通道核识别并组合输入中的模式(对上面的多通道输出进行加权组合)
# 1×1卷积层
#   k_h=k_w=1是一个受欢迎的选择, 它不识别空间模式, 只是融合通道
#   相当于输入形状n_h n_w × c_i, 权重为c_i × c_o的全连接
# 二维卷积层
#   输入X: c_i × n_h × n_w
#   核W: c_o × c_i × k_h × k_w
#   偏差B: c_o × c_i
#   输出Y: c_o × m_h × m_w
#   计算复杂度(浮点计算数FLOP) O(c_i c_o k_h _kW m_h m_w)
#       c_i=c_o=100, k_h=h_w=5, m_h=m_w=64  => 1G FLOP
#   10层, 1M样本, 10PFlops (CPU: 0.15TF=18h, GPU: 12TF=14min)
# 总结
#   输出通道数是卷积层的超参数
#   每个输入通道有独立的二维卷积核, 所有通道结果相加得到一个输出通道结果
#   每个输出通道有独立的三维卷积核

import torch
from d2l import torch as d2l

# 实现多输入通道互相关运算
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))

# 计算多个通道的输出的互相关函数
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
print(corr2d_multi_in_out(X, K))

# 1×1的卷积
def corr2d_multi_in_out_1x1(X, K):  # 全连接计算
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6