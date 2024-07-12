# 池化层: 作用在卷积层的输出上
#   卷积层对位置敏感
#       检测垂直边缘, 只要有1个像素偏移, 就会导致0输出
#       而一般边缘在图像中不是那么规矩, 可能会有弯曲的地方, 而且相机的抖动或者物体的移动都会导致边缘发生变化, 所以需要一定的平移不变性(输入稍微有一点改动, 输出不会有太大的变化, 减低卷积核对位置的敏感程度)
#   需要一定程度的平移不变性
#       照明, 物体位置, 比例, 外观等等因图像而异
# 二维最大池化
#   返回滑动窗口中的最大值
#   2×2最大池化可以允许1像素的移位, 并可以增加一点模糊效果
# 填充, 步幅和多个通道
#   池化层与卷积层类似, 都具有填充和步幅
#   没有可学习的参数
#   在每个输入通道应用池化层以获得相应的输出通道
#   输出通道数 = 输入通道数
# 平均池化层
#   最大池化层: 每个窗口中最强的模式信号
#   平均池化层: 将最大池化层中的"最大"操作替换为"平均"
# 总结
#   池化层返回窗口中最大或平均值
#   缓解卷积层对未知的敏感性
#   同样有窗口大小, 填充, 步幅作为超参数

import torch
from torch import nn
from d2l import torch as d2l

# 实现池化层的正向传播, 步长为1, 没有填充
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))

# 填充和步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)
pool2d = nn.MaxPool2d(3)    # 默认步幅和窗口大小相同, 默认不填充
print(pool2d(X))
# 手动设定填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
# 设定一个任意大小的矩形汇聚窗口, 并分别设定填充和步幅的高度和宽度
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(1, 1))
print(pool2d(X))

# 多个通道
# 池化层在每个通道上单独计算
X = torch.cat((X, X + 1), 1)    #stack是升维拼接, cat是等维拼接
print(X)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
