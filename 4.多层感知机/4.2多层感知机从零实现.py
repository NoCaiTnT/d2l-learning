import torch
from torch import nn
from d2l import torch as d2l

# 为什么参数的初始值不是0或者1或者全相等? !!!!!
#   在神经网络中, 如果将权值初始化为0, 或者其他统一的常量, 会导致后面的激活单元具有相同的值, 所有的单元相同意味着它们都在计算同一特征, 网络变得跟只有一个隐含层节点一样, 这使得神经网络失去了学习不同特征的能力
#   对称性问题：如果所有的权重初始化为相同的值（比如0），在反向传播算法中，所有权重的梯度将相等。这会导致所有的权重在更新过程中保持相等，从而使得网络无法学习到任何有用的特征。
#   梯度消失：如果所有的权重都初始化为0，那么在反向传播过程中，所有的梯度也将为0，这会导致梯度下降无法正常工作。这种情况下，网络将无法学习复杂的非线性关系。
#   局部最小值：如果所有的权重都初始化为相同的值，那么所有的神经元将以相同的方式响应相同的输入。这可能会使得网络陷入局部最小值，并且无法找到全局最优解。
# 权重初始化全相等为一个常数为什么不行？
#   在这种情况下, 在前向传播期间, 两个隐藏单元采用相同的输入和参数, 产生相同的激活, 该激活被送到输出单元。在反向传播期间, 根据参数W对输出单元进行微分, 得到一个梯度, 其元素都取相同的值。
#   因此, 在基于梯度的迭代(例如小批量随机梯度下降)之后, W的所有元素仍然采用相同的值。这样的迭代永远不会打破对称性, 我们可能永远也无法实现网络的表达能力。隐藏层的行为就好像只有一个单元。




# 导入数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

# ReLU激活函数
def relu(X):
    a = torch.zeros_like(X) # 生成数据类型, 形状一样, 大小全为0
    return torch.max(X, a)

# 模型
def net(X):
    X = X.reshape((-1, num_inputs))     #拉成一维矩阵
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return (H @ W2 + b2)

# 交叉熵损失
loss = nn.CrossEntropyLoss(reduction='none')

# 训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 预测
d2l.predict_ch3(net, test_iter)