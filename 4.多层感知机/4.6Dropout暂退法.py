# 动机
#   一个好的模型需要对输入数据有扰动鲁棒性
#       使用有噪音的数据等价于Tikhonov正则
#       丢弃法: 是在层之间加入噪音
# 无偏差的加入噪音
#   在每一层对x加入噪音得到x', 但是不改变期望 E[x'] = x
#   丢弃法对每个元素进行如下扰动
#       x_i' = 0            以概率p进行丢弃
#            = x_i / (1-p)  否则
# dropout使用
#   通常将dropout作用在隐藏全连接层的输出上(在训练上)
#       h = σ(w1x + b1), h‘ = dropout(h), o = w2h' + b2, y = softmax(o)
# 推理中的丢弃法
#   dropout是一个正则项(l1, l2等也是)
#   正则项只在训练中使用: 他们影响模型参数的更新
#   在推理过程中, dropout直接返回输入: h = dropout(h)
#   保证确定性的输出
# 总结
#   dropout将一些输出项随机置0来控制模型复杂度
#   常用于MLP的隐藏层输出上(很少用在CNN上)
#   丢弃概率是控制模型复杂度的超参数

import torch
from torch import nn
from d2l import torch as d2l

if __name__ == "__main__":
    # 从零实现
    # 定义dropout函数
    # def dropout_layer(X, dropout):
    #     assert 0 <= dropout <= 1
    #     # 在本情况中，所有元素都被丢弃
    #     if dropout == 1:
    #         return torch.zeros_like(X)
    #     # 在本情况中，所有元素都被保留
    #     if dropout == 0:
    #         return X
    #     mask = (torch.rand(X.shape) > dropout).float()      # 为什么不是从中选择几个设置为0(X[maxk] = 0)而是使用乘法？ 因为乘法的计算开销小, 计算快
    #     return mask * X / (1.0 - dropout)
    #
    # # # 测试dropout函数
    # # X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    # # print(X)
    # # print(dropout_layer(X, 0.))
    # # print(dropout_layer(X, 0.5))
    # # print(dropout_layer(X, 1.))
    #
    # # 定义模型, 两层隐藏层, 每个隐藏层包含256个单元
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5
    # class Net(nn.Module):
    #     def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
    #         super(Net, self).__init__()
    #         self.num_inputs = num_inputs
    #         self.training = is_training
    #         self.lin1 = nn.Linear(num_inputs, num_hiddens1)
    #         self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
    #         self.lin3 = nn.Linear(num_hiddens2, num_outputs)
    #         self.relu = nn.ReLU()
    #
    #     def forward(self, X):
    #         H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
    #         # 只有在训练模型时才使用dropout
    #         if self.training == True:
    #             # 在第一个全连接层之后添加一个dropout层
    #             H1 = dropout_layer(H1, dropout1)
    #         H2 = self.relu(self.lin2(H1))
    #         if self.training == True:
    #             # 在第二个全连接层之后添加一个dropout层
    #             H2 = dropout_layer(H2, dropout2)
    #         out = self.lin3(H2)
    #         return out
    # net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    #
    # 训练和测试
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    # 简洁实现
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        # 在第一个全连接层之后添加一个dropout层
                        nn.Dropout(dropout1),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        # 在第二个全连接层之后添加一个dropout层
                        nn.Dropout(dropout2),
                        nn.Linear(256, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    # 训练
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)