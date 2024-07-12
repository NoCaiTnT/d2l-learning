# 全连接卷积神经网络(FCN, fully convolutional network)
#   FCN是用深度神经网络来做语义分割的奠基性工作
#   它用转置卷积层来替换CNN最后的全连接层, 从而可以实现每个像素的预测
#   在CNN(去掉全局平均池化层和全连接层)后面加入一个 1×1的卷积层(降低维度, 减少通道数) 再加一个转置卷积层
#   图像高宽: 224×224 ->(CNN) 7×7 ->(1×1 Conv) 7×7 ->(转置卷积) 224×224×K   (对每个像素类别的预测值, 存在通道里, 有K类就有K个通道)

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torchvision.models import ResNet18_Weights

# 使用预训练的ResNet18来提取图像特征(做CNN)
pretrained_net = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print(list(pretrained_net.children())[-3:])

# 创建一个全卷积网络
# 去掉ResNet最后的全局平均池化层和全连接层
net = nn.Sequential(*list(pretrained_net.children())[:-2])
# net的前向传播将输入的高和宽减小至原来的1/32
X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape)

# 使用1×1的卷积层将输出通道数转换为Pascal VOC2012数据集的类数（21类）
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))  # 这里输出通道可以取512到21之间的任意数, 这里直接取21是为了计算简单, 为了让下面图像放大的计算量减少
# 将特征图的高度和宽度增加32倍，从而将其变回输入图像的高和宽
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))

# 使用双线性插值 初始化转置卷积层
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

# 双线性插值的上采样实验
# 构造一个将输入的高和宽放大2倍的转置卷积层，并将其卷积核用bilinear_kernel函数初始化
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
# 读取图像X，将上采样的结果记作Y
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
# 输出
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))
d2l.plt.show()
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img)
d2l.plt.show()

# 全卷积网络用双线性插值的上采样初始化转置卷积层
# 对于1×1卷积层，我们使用Xavier初始化参数。
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# 读取数据集
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

# 训练
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 预测
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

# 将预测类别映射回它们在数据集中的标注颜色
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]

# 读取几张较大的测试图像，并从图像的左上角开始截取形状为320×480的区域用于预测
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
d2l.plt.show()