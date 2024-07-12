# 问题
#   给定(32×32)的输入图像
#   应用5×5大小的卷积核
#       第1层得到的输出大小为28×28
#       第7层得到的输出大小为4×4
#   更大的卷积核可以更快地减小输出大小
#       形状从n_h × n_w减少到(n_h - k_h + 1) × (n_w - k_w + 1)
# 填充
#   在输入周围添加额外的行/列, 如0
#   填充p_h行和p_w列, 输出形状为: (n_h - k_h + p_h + 1) × (n_w - k_w + p_w + 1)
#   通常取p_h = k_h - 1, p_w = k_w - 1, 即输出和输入的大小一样
#       当k_h为奇数: 在上下两侧填充p_h/2
#       当k_h为偶数: 在上侧填充p_h/2(向上取整), 在下侧填充p_h/2(向下取整), 即上侧比下侧多填一行
# 步幅
#   填充减小的输入大小与层数线性相关
#       给定输入大小224×224, 在使用5×5卷积核的情况下, 需要44层将输出降低到4×4
#       需要大量计算才能得到较小输出
#   步幅是指行/列的滑动步长
#       例: 高度3, 宽度2的步幅
#   给定高度s_h和宽度s_w的步幅, 输出形状为: [(n_h - k_h + p_h + s_h)/s_h](向下取整) × [(n_w - k_w + p_w + s_w)/s_w](向下取整)
#   如果p_h = k_h - 1, p_w = k_w - 1, 则[(n_h + s_h - 1)/s_h](向下取整) × [(n_w + s_w - 1)/s_w](向下取整)
#   如果输入高度和宽度可以被步幅整除, 则为: (n_h/s_h)/(n_w/s_w)
# 总结
#   填充和步幅是卷积层的超参数
#   填充在输入周围添加额外的行/列, 来控制输出形状的减小
#   步幅是每次滑动核窗口时的行/列的步长, 可以成倍的减少输出形状

import torch
from torch import nn

# 填充
# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)     # 加入通道数和图像个数, 都为1
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # 输入输出的通道数都为1
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)

# 填充不同的宽度和高度
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

# 步幅
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

# 复杂的例子
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
