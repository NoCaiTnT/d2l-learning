# 网络中的网络(NiN)
# 全连接层的问题(在LeNet, AlexNet, VGG中最后都用到了几个很大的全连接层)
#   卷积层需要较少的参数: c_i × c_o × k²
#   但卷积层后的第一个全连接层的参数很大:
#       LeNet: 16×5×5×120 = 48k
#       AlexNet: 256×5×5×4096 = 26M
#       VGG: 512×7×7×4096 = 102M
#   参数太大导致的问题:
#       需要很大的内存
#       占用很多的计算带宽
#       很容易过拟合
# NiN块
#   一个卷积层后跟两个全连接层
#       步幅1, 无填充, 输出形状跟卷积层输入一样
#       起到全连接层的作用(对每个像素做全连接层)
# NiN架构
#   无全连接层
#   交替使用NiN块和步幅为2的最大池化层(使用4次, 第4次直接接最后的全局平均池化)
#       逐步减小高宽和增大通道数
#   最后使用全局平均池化层得到输出
#       全局平均池化层: 池化层的高宽等于输入的高宽, 即对每个通道取平均, 得到一个元素
#       其输入通道数是类别数
# 总结
#   NiN块使用卷积层 + 两个1×1卷积层
#       后者对每个像素增加了非线性性
#   NiN使用全局平均池化层来替代VGG和AlexNet中的全连接层
#       不容易过拟合, 更少的参数个数

# 实现
import torch
from torch import nn
from d2l import torch as d2l

# NiN网络块, 1个普通卷积层 + 2个1×1卷积层(充当带有ReLU激活函数的逐像素全连接层)
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

# 定义网络结构
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())

# 查看网络结构及块的输出大小
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# 训练模型
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
