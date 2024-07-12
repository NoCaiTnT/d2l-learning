# 序列模型
#   实际中很多数据是有时序结构的
#   电影的评价随时间变化而变化
#       拿奖后评分上升, 直到奖项被遗忘
#       拿了很多好电影后, 人们的期望变高
#       季节性: 贺岁片, 暑期档
#       导演, 演员的负面报道导致评分变低
# 更多例子
#   音乐, 语言, 文本和视频都是连续的
#       标题"狗咬人"远没有"人咬狗"那么令人惊讶
#   大地震发生后, 很可能会有几次较小的余震
#   人的互动是连续的, 从网上吵架可以看出
#   预测明天的股价要比填补昨天遗失的股价更难
# 统计工具
#   在时间t观察到x_t, 那么得到T个不独立的随机变量(x_1, ..., x_T)~p(X)
#   使用条件概率展开: p(a,b) = p(a)p(b|a)=p(b)p(a|b)
#   p(X)=p(x_1) p(x_2|x_1) p(x_3|x_1,x_2) ... p(x_T|x_1, ..., x_T-1)    前向: 知道过去推未来
#   p(X)=p(x_T) p(x_T-1|x_T) p(x_T-2|x_T-1,X_T) ... p(x1|x_2, ..., x_T) 反向: 知道未来推过去
# 序列模型 前向: 知道过去推未来
#   对条件概率建模: p(x_t|x1, ..., x_t-1) = p(x_t|f(x1, ..., x_t-1))
#   f()对见过的数据建模, 也称自回归模型
# 方案A - 马尔科夫假设 前向: 知道过去推未来
#   假设当前数据只跟τ个过去数据点相关
#   p(x_t|x1, ..., x_t-1) = p(x_t|x_t-τ, ..., x_t-1) = p(x_t|f(x_t-τ, ..., x_t-1))
#   假如在过去数据上训练一个MLP模型(输入为固定长度)
# 方案B - 潜变量模型 前向: 知道过去推未来
#   引入潜变量h_t来表示过去的信息h_t = f(x1, ..., x_t-1)
#       这样x_t = p(x_t | h_t)
#   h → h' →
#   ↓ ↗ ↓  ↗
#   x → x' →
#   相当于拆成两个模型
#       1. 新的潜变量h', 通过上一个潜变量h和上一个值x 计算新的潜变量h'
#       2. 新的值x', 通过新的潜变量h'和上一个值x 计算新的值x'
# 总结
#   时序模型中, 当前数据跟之前观察到的数据相关
#   自回归模型使用自身过去数据来预测未来
#   马尔科夫模型假设当前只跟最近少数数据相关
#   潜变量模型使用潜变量来概括历史信息

# 使用正弦函数和一些可加性噪声来生成序列数据, 时间步为1, 2, ..., 1000
import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

# 使用马尔科夫假设
# 将数据映射为数据对y_t = x_t 和 X_t = [x_t-τ, ..., x_t-1]
# τ为4, 一共1000个点, 每四个预测一个, 所以倒数第五个到倒数第二个 预测倒数第一个, 所以有1000-4组数据
# 第一组的特征为1,2,3,4, 第二组特征为2,3,4,5,, 竖着来看是1,2,3...和2,3,4..., 所以一共4列, 分别为1-996, 2-997, 3-998, 4-999
# 第一组的预测值为5, 第二组为6, 因此预测值为5-1000
# 类似滑动窗口
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练 使用前600个样本训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

# 使用一个两个全连接层的多层感知机
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

# 训练函数
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 预测
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.show()

# 多步预测
# 使用预测出来的数据 作为特征 继续预测
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()
# 多步预测结果相差很远
# 原因: 每次预测都有误差, 误差是不断的叠加的
# k步预测, 给定一组特征点, 预测后面k个点
# 如1步预测, 给4个特征点预测1个点, 很准
# 如64步预测, 给4个特征点预测64个点, 不准(后续的特征点是基于自身预测出来的点, 而不是真实特征点)

max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.show()

# MLP的问题: 很难预测很远的未来