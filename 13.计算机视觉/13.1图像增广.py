# CES上的真实故事
#   有一家做智能售货机的公式, 发现他们的演示机器在现场的效果很差, 因为现场有: 不同的色温, 桌面的灯光反射不一样
#   他们连夜现场收集了数据, 训练了一个新的模型, 同时买了一块新桌布
# 数据增强
#   增加一个已有数据集, 使得有更多的多样性
#       在语言里面加入各种不同的背景噪音
#       改变图片的颜色和形状
# 使用增强数据训练
#   原始数据 ->(在线随机生成) 增强后的数据 -> 训练
# 翻转
#   左右翻转
#   上下翻转(不总是可行)
# 切割
#   从图片中切割一块, 然后变形到固定形状
#       随机高宽比(例如[3/4, 4/3])
#       随机大小(例如[8%, 100%])  切的图片占原图的百分之多少
#       随机位置
# 颜色
#   改变色调, 饱和度, 明亮度(例如[0.5, 1.5])
# 总结
#   数据增广通过变形数据来获取多样性从而使得模型泛化性更好
#   常见的图片增广包括翻转、切割、变形

import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

# 对img做aug数据增广, 重复2*4次, 缩放1.5倍
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()

# 随机水平翻转(左右翻转)
apply(img, torchvision.transforms.RandomHorizontalFlip())
# 随机上下翻转
apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机裁剪
# 最终的输出为200*200大小, 裁剪出来的图片大小为原图的0.1-1之间, 高宽比为0.2-2
shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 随机改变图像的亮度
# 亮度(上下50%), 对比度, 饱和度, 色调
apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))
# 改变色调
apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))
# 一起随机改变(上下50%)
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 结合多种图像增广方法
# 随机翻转 -> 随机改变颜色亮度等 -> 随机裁剪
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

# 使用图像增广进行训练
# CIFAR10: ImageNet里面采样了10个类, 3通道的彩色RGB图像, 图片尺寸为32×32, 每个类别有6000个图像，数据集中一共有50000张训练图片和10000张测试图片
all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
d2l.plt.show()

# 只使用最简单的随机左右翻转
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

# 定义一个辅助函数, 以便读取图像和应用图像增广
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train, transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

# 定义一个函数, 使用多GPU对模型进行训练和评估
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    # 用多GPU进行小批量训练
    # 判断X是否是一个列表, 将训练数据放到GPU0上
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    # 用多GPU进行模型训练
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    d2l.plt.show()

# 定义train_with_data_aug函数, 使用图像增广来训练模型
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

# 使用随机左右翻转训练
train_with_data_aug(train_augs, test_augs, net)

# 不使用数据增广
train_with_data_aug(test_augs, test_augs, net)
# 训练精度变高, 测试精度变化不大, 过拟合更加严重了