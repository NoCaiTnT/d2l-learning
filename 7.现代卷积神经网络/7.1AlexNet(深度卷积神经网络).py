# 其他机器学习: 核方法(支持向量机SVG)
#   特征提取
#   选择核函数来计算相似性
#   凸优化问题
#   漂亮的定理
# 几何学(MVG)
#   抽取特征
#   描述几何(例如多相机)
#   (非)凸优化
#   漂亮的定理
#   如果假设满足了, 效果非常好
# 特征工程
#   特征工程师关键
#   特征描述子: SIFT, SURF
#   视觉词袋(聚类)
#   最后用SVM
# ImageNet
#   图片: 自然物体的彩色图片
#   大小: 469×387
#   样本数: 1.2M
#   类数: 1000
# AlexNet
#   更深更大的LeNet
#   主要改进: 丢弃法(模型大小的控制), ReLU(梯度更大, 0点导数好), MaxPooling(输出梯度更大, 更容易训练)
#   计算机视觉方法论的改变
#       之前: 人工特征提取 -> SVM(标准机器学习)
#       之后: 通过CNN学习特征 -> Softmax回归
# AlexNet架构
#   输入: 3×224×224
#   11×11 Conv (96), stride 4
#   3×3 MaxPool, stride 2
#   5×5 Conv (256), pad 2
#   3×3 MaxPool, stride 2
#   3×3 Conv (384), pad 1
#   3×3 Conv (384), pad 1
#   3×3 Conv (384), pad 1
#   3×3 MaxPool, stride 2
#   Dense (4096)    # Dense 全连接层
#   Dense (4096)
#   Dense (1000)
#   更多细节
#       激活函数从sigmoid变为ReLU(减缓梯度消失)
#       隐藏全连接层后加入了丢弃层
#       数据增强
# 复杂度
#                     参数个数                   FLOP
#               AlexNet     LeNet       ALexNet     Lenet
#   Conv1       35K         150         101M        1.2M
#   Conv2       614K        2.4K        415M        2.4M
#   Conv3-5     3M                      445M
#   Dense1      26M         0.48M       26M         0.48M
#   Dense2      16M         0.1M        16M         0.1M
#   Total       46M         0.6M        1G          4M
#   Increase    11x         1x          250x        1x
# 总结
#   AlexNet是更大更深的LeNet, 10x参数个数, 260x计算复杂度
#   新进入了丢弃法, ReLU, 最大池化层, 数据增强
#   AlexNet赢下2012ImageNet竞赛后, 标志着新一轮神经网络热潮的开始
# 实现
import torch
from torch import nn
from d2l import torch as d2l

# 定义AlexNet网络模型
net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    # 使用的是fashion_mnist数据集, 因此输入维度为1, 输出维度为10
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))

# 查看层结构及输出大小
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 读取数据, 并将数据放缩为AlexNet的数据集大小
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# 训练
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

