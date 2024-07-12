# 残差块
#   串联一个层改变函数类, 希望能扩大函数类
#   残差块加入快速通道(右边)来得到f(x) = x + g(x)的结构
# ResNet块细节
#   x -> 3×3 Conv -> BN -> ReLU -> 3×3 Conv -> BN -> ReLU
#     ↓->              (1×1 Conv)                ->↑
# ResNet块
#   高宽减半ResNet块(步幅2)
#   后接多个高宽不变的ResNet块
# ResNet架构
#   类似VGG和GoogLeNet的总体架构
#   但替换为了ResNet块
#   7×7 Conv + BN + 3×3 Max Pooling
#   4个ResNet
#   global average pool
# 总结
#   残差块使得很深的网络更加容易训练, 甚至可以训练一千层的网络
#   残差网络对随后的深层神经网络设计产生了深远影响, 无论是卷积类网络还是全连接类网络
# 避免了梯度消失: 将乘法运算变成加法运算(残差链接)

# 实现
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 残差网络
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 查看输入和输出形状
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

# 增加通道数, 减半输出的高和宽
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)

# ResNet模型, 5层
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 残差块
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

# 查看不同模块的输入输出
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# 训练
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
