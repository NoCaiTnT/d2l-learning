import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 训练误差和泛化误差
#   训练误差: 模型在训练数据上的误差
#   泛化误差: 模型在新数据上的误差
# 验证数据集和测试数据集
#   验证数据集: 一个用来评估模型好坏的数据集, 不要跟训练数据混在一起, 用来评估超参数的好坏
#   测试数据集: 只用一次的数据集, 不能用来调超参数
# K-折交叉验证(cross validation) 用于选择超参数
#   在没有足够多数据时使用
#   算法:
#       将训练数据分割成K块
#       for i= 1, ..., K
#           使用第i块作为验证数据集, 其余作为训练数据集
#       报告K个验证集误差的平均
#       K一般取5或10(即需要训练K次)
# 总结
#   训练数据集: 训练模型参数
#   验证数据集: 选择模型超参数
#   非大数据集上通常使用k-折交叉验证
# 过拟合和欠拟合
#   模型容量低, 数据简单, 正常结果
#   模型容量低, 数据复杂, 欠拟合
#   模型容量高, 数据简单, 过拟合
#   模型容量高, 数据复杂, 正常结果
# 模型容量
#   拟合各种函数的能力
#   低容量的模型难以拟合训练数据
#   高容量的模型可以记住所有的训练数据
#   模型容量越大, 训练误差越小, 泛化误差先变小后变大
# 估计模型容量
#   难以在不同种类的算法之间比较: 例如树模型和神经网络
#   给定一个模型种类, 将有两个主要因素: 参数的个数, 参数值的选择范围
# VC维
#   对于一个分类模型, VC等于一个最大的数据集大小, 不管如何给定标号, 都存在一个模型来对它进行完美分类
# 线性分类器的VC维
#   2维输入的感知机, VC维 = 3, 即能够分类任何三个点, 但不是四个(XOR)
#   支持N维输入的感知机的VC维是N+1
#   一些多层感知机的VC维是O(N log_2 N)
# VC维的用处
#   提供为什么一个模型好的理论依据: 衡量训练误差和泛化误差之间的间隔
#   但在深度学习中很少使用:
#       衡量不是很准确
#       计算深度学习模型的VC维很困难
# 数据复杂度
#   样本个数, 每个样本的元素个数, 时间/空间结构, 多样性(如类别数)
# 总结
#   模型容量需要匹配数据复杂度, 否则可能导致欠拟合和过拟合
#   统计机器学习提供数学工具来衡量模型复杂度
#   实际中一般靠观察训练误差和验证误差

if __name__ == "__main__":
    # 拟合 y = 5 + 1.2x - 3.4*x*x/2! + 5.6*x*x*x/3!
    # 生成数据集和标签, 加入误差项, 误差项符合正态分布, 均值为0, 方差为0.1²
    max_degree = 20  # 多项式的最大阶数
    n_train, n_test = 100, 100  # 训练和测试数据集大小
    true_w = np.zeros(max_degree)  # 分配大量的空间
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
    # labels的维度:(n_train+n_test,)
    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    # NumPy ndarray转换为tensor
    true_w, features, poly_features, labels = [torch.tensor(x, dtype=
        torch.float32) for x in [true_w, features, poly_features, labels]]

    print(features[:2], poly_features[:2, :], labels[:2])

    # 定义评估函数
    def evaluate_loss(net, data_iter, loss):  #@save
        """评估给定数据集上模型的损失"""
        metric = d2l.Accumulator(2)  # 损失的总和,样本数量
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric.add(l.sum(), l.numel())
        return metric[0] / metric[1]

    # 定义训练函数
    def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
        loss = nn.MSELoss(reduction='none')
        input_shape = train_features.shape[-1]
        # 不设置偏置，因为我们已经在多项式中实现了它
        net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
        batch_size = min(10, train_labels.shape[0])
        train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)),
                                    batch_size)
        test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)),
                                   batch_size, is_train=False)
        trainer = torch.optim.SGD(net.parameters(), lr=0.01)
        animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                                xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                                legend=['train', 'test'])
        for epoch in range(num_epochs):
            d2l.train_epoch_ch3(net, train_iter, loss, trainer)
            if epoch == 0 or (epoch + 1) % 20 == 0:
                animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                         evaluate_loss(net, test_iter, loss)))
        d2l.plt.show()  # add
        print('weight:', net[0].weight.data.numpy())

    # 正常: 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
    train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

    # 欠拟合: 从多项式特征中选择前2个维度，即1和x
    train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

    # 过拟合: 从多项式特征中选取所有维度
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)
