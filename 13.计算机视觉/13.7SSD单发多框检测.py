# SSD实现

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 预测锚框的类别
# 输入的通道数, 锚框的数量, 类别的数量(预测类+背景类), 对每一个锚框都进行预测
# 预测的个数: 像素总数 × 每个像素的锚框数 × (预测的类别总数+背景类)
# 输出: 和输入图像大小一样, 通道数就是该像素对应的锚框数
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

# 边界框预测
# 预测与真实框的偏移, 偏移用4个值表示(13.4.3.2有计算公式)
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# 连接多尺度的预测
def forward(x, block):
    return block(x)

# 举例说明每个层的输出尺度都不一样
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape, Y2.shape)   # Y1: (2, 5*(10+1), 20, 20)   Y2: (2, 3*(10+1), 10, 10)

# 通道数放到最后, 将4维向量变成2维(把后面三个维度拉成1维)
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)
# 将不同维度的预测直接叠起来
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
print(concat_preds([Y1, Y2]).shape) # 25300 = 20*20*55 + 10*10*33

# 高和宽减半块, 改变通道数
# 两个卷积+一个池化
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)

# 基本网络块
# 从原始图像抽取特征, 直到第一次抽取锚框
# 3个高宽减半, 通道数最终变成64
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

# 完成的单发多框检测模型由五个模块组成
def get_blk(i):
    if i == 0:  # block 1   (2,3,256,256) -> (2,64,32,32)
        blk = base_net()
    elif i == 1:    # block 2   (2,3,32,32) -> (2,128,16,16)
        blk = down_sample_blk(64, 128)
    elif i == 4:    # block 5   (2,128,4,4) -> (2,128,1,1)
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:   # block 3 4         (2,128,16,16) -> (2,128,8,8) -> (2,128,4,4)
        blk = down_sample_blk(128, 128)
    return blk

# 定义每个块的前向计算
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    # 根据该层的特征大小, 生成锚框
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

# 定义超参数
# 一共5层, 每层的放缩, 以及高宽比
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1        # 生成4个锚框

# 定义完整的模型
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]   # 每一层的输出通道数
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            # 相当于self.blk_1 = get_blk(1)的简洁写法
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        # 算出5个阶段的锚框及其预测的类别, 偏移, 并记录
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        # 将5个阶段的结果叠起来
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

# 创建一个模型实例, 然后使用它执行前向计算
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)     # (1, 5444, 4)  5444 = 32*32*4 + 16*16*4 + 8*8*4 + 4*4*4 + 1*1*4
print('output class preds:', cls_preds.shape)   # (32, 5444, 2)
print('output bbox preds:', bbox_preds.shape)   # (32, 21776) 21776 = 5444 * 4

# 读取香蕉检测数据集
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

# 初始化参数 定义优化算法
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# 定义损失函数和评价函数
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)      # 若是背景, mask=0, 就不算偏移了
    return cls + bbox

# 评价指标, 类别预测一致的个数
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())
# 评价指标, 预测框偏移差值的和, 只关注非背景锚框
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# 训练模型
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on ' f'{str(device)}')

# 预测
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

# 通过锚框和预测的偏移得到实际预测的预测框
# 然后根据类别的置信度通过NMS只保留置信度最高的预测框
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)

# 筛选出所有置信度不低于0.9的边界框作为最终输出
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
d2l.plt.show()