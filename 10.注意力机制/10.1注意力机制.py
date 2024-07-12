# 心理学
#   动物需要在复杂环境下有效关注值得注意的点
#   心理学框架: 人类根据随意线索和不随意线索选择注意点
#   随意: 有目的, 不随意: 没目的

# 注意力机制
#   卷积, 全连接, 池化层都只考虑不随意线索
#   注意力机制显示的考虑随意线索
#       随意线索被称之为查询(query), 例如想喝咖啡
#       每个输入是一个值(value)和不随意线索(key)的对(环境)
#       通过注意力池化层来有偏向的选择某些输入

# 非参注意力池化层
#   给定数据(x_i, y_i), i=1,...,n
#   平均池化是最简单的方案： f(x) = 1/n ∑_i y_i
#   更好的方案是60年代提出来的Nadaraya-Watson核回归
#       f(x) = ∑_i^n y_i K(x, x_i) / ∑_{j=1}^n K(x, x_j)
#         ↑           ↑                           ↑
#       query        value                       key

# Nadaraya-Watson核回归
#   使用高斯核: K(u) = 1/sqrt(2π) exp(-u^2/2)
#   那么f(x) = ∑_i^n y_i softmax(-1/2 (x-x_i)^2)

# 参数化的注意力机制
#   在之前基础上引入可以学习的w
#   f(x) = ∑_i^n y_i softmax(-1/2 ((x-x_i)w)^2)
#   通过学习w, 可以学习到不同的注意力权重

# 总结
#   心理学认为人通过随意线索和不随意线索选择注意点
#   注意力机制中, 通过query(随意线索)和key(不随意线索)来有偏向性的选择输入
#       可以一般的写作f(x) = ∑_i α(x,x_i)y_i, 这里的α(x,x_i)是注意力权重
#       早在60年代就有非参数的注意力机制
#       接下来会介绍多个不同的权重设计

# 注意力汇聚: Nadaraya-Watson核回归的实现
import torch
from torch import nn
from d2l import torch as d2l

# 生成数据集
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)
# 真实函数
def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)
print(n_test)

# 绘制
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()

y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)

# 非参数注意力汇聚
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

# 注意力权重
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                    xlabel='Sorted training inputs', ylabel='Sorted testing inputs')

# 带参数注意力汇聚 假定两个张量的形状分别是(n,a,b)和(n,b,c), 它们的批量矩阵乘法输出的形状是(n,a,c)
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(torch.bmm(X, Y).shape)

# 使用小批量矩阵乘法来计算小批量数据中的加权平均值
wights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
print(torch.bmm(wights.unsqueeze(1), values.unsqueeze(-1)))

# 带参数的注意力汇聚
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape(-1, keys.shape[1])
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

# 将训练数据集转换为键和值
X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

# 训练带参数的注意力汇聚模型
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

# 预测结果绘制
keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

# 曲线在注意力权重较大的区域变得更不平滑
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
