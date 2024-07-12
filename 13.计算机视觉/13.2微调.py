# 网络架构
#   一个神经网络一般可以分成两块: 特征抽取, 线性分类器(softmax回归)
#       特征抽取将原始像素变成容易线性分割的特征
#       线性分类器来做分类
# 微调
#   在源数据集上训练好的特征抽取部分, 可能仍然可以对我数据集做特征抽取
#   而线性分类器不能直接使用, 因为标号可能变了
# 微调中的权重初始化(使用一样的模型架构)
#   把预训练模型中特征抽取部分的权重复制到自己的模型当中
#   最后一层的输出层(线性分类器)随机初始化权重
# 训练
#   是一个目标数据集上的正常训练任务, 但使用更强的正则化(已经学的差不多了)
#       使用更小的学习率
#       使用更少的数据迭代
#   源数据集远复杂于目标数据集, 通常微调效果更好
# 重用分类器权重
#   源数据集可能也有目标数据中的部分标号
#   可以使用预训练模型分类器中对应标号的对应向量(权重)来做初始化
# 固定一些层
#   神经网络通常学习有层次的特征表示
#       低层次的特征更加通用
#       高层次的特征则跟数据集相关
#   可以固定底部一些层的参数, 不参与更新
#       更强的正则
# 总结
#   微调通过使用大数据集上得到的预训练好的模型来初始化模型权重来完成提升精度
#   预训练模型质量很重要
#   微调通常速度更快, 精度更高

import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 下载数据集
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

# 读取训练集和测试集
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 显示前8个正类样本图片和最后8张负类样本图片
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
d2l.plt.show()

# 使用数据增广
# 使用RGB通道的均值和标准差，以标准化每个通道, 这组参数从ImageNet数据集获得
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406],  # 对图像做均值3通道
    [0.229, 0.224, 0.225])  # 对图像做标准差3通道

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),      # 随机裁剪
    torchvision.transforms.RandomHorizontalFlip(),      # 随机水平(左右)翻转
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# 定义和初始化模型
pretrained_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
# 打印网络结构
print(pretrained_net)

# 定义自己的微调模型
finetune_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

# 微调模型
# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        # 最后一层是随机权重, 学习率设置为10倍, 其它层不变
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 微调
# train_fine_tuning(finetune_net, 5e-5)

# 对比, 重新训练
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)

# 首先你要构建一个和你所说的外部模型一样结构的模型（你想要载入的那部分要一致，无需载入的部分如最后的输出层倒无所谓），然后加载那个模型预训练的state_dict，要注意的是你构建的模型与预训练的模型各层的命名要一致，否则load_state_dict会报错。load_state_dict中参数设置strict=False，仅加载一致部分（如下代码）
#
# # 构建模型
# # Build your model
#
# # 加载预训练的state_dict
# pretrained_state_dict = torch.load(pretrained_path)
# # 检验参数数量，确保能匹配
# matched_state_dict = {k: v for k, v in pretrained_state_dict.items()
#                       if model.state_dict()[k].numel() == v.numel()}
# # 不严格载入，加载模型中一致的部分
# model.load_state_dict(matched_state_dict, strict=False)
#
# # 后续微调
# # To be implemented