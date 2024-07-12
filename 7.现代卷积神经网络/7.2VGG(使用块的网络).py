# VGG
#   AlexNet比LeNet更深更大来得到更好的精度, 能不能再更深和更大?
#   选项
#       更多的全连接层(太贵)
#       更多的卷积层
#       将卷积层组合成块
# VGG块
#   深 vs 宽
#       5×5卷积
#       3×3卷积
#       深但窄效果更好(同样的计算开销)
#   VGG块
#       3×3卷积(填充1), 用n层, m通道
#       2×2最大池化层(步幅2)
# VGG架构
#   多个VGG块后接全连接层
#   不同次数的重复块得到不同的架构VGG-16, VGG-19
#   对AlexNet的改进相当于将AlexNet前面不规则的卷积层抽象为VGG块, 然后重复使用
# 进度
#   LeNet(1995)
#       2卷积+池化层
#       2全连接层
#   AlexNet
#       更大更深
#       ReLU, Dropout, 数据增强
#   VGG
#       更大更深的AlexNet(重复的VGG块)
# 总结
#   VGG使用可重复使用的卷积块来构建深度卷积神经网络
#   不同的卷积块个数和超参数可以得到不同复杂度的变种

# 实现
import torch
from torch import nn
from d2l import torch as d2l

# 构造卷积块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

# 网络结构
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1         # 输入是灰度图是1维
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = vgg(conv_arch)

# 输出网络结构及每层大小
X = torch.randn(size=(1, 1, 96, 96))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

# 训练
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
