# 小批量随机梯度下降
#   计算单样本的梯度难完全利用硬件资源
#   小批量随机梯度下降在时间t采用一个随机子集I_t ⊆ {1,2,...,n}使得|I_t|=b
#       x_t = x_{t-1} - η_t / b * Σ_{i∈I_t} ∇l_i(x_{t-1})
#   通用这是一个无偏的近似, 但降低了方差
#       1/b * Σ_{i∈I_t} ∇l_i(x) 的期望 = ∇f(x) 的期望

import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

timer = d2l.Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)

# 逐元素计算A=BC
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
print('单元素计算时间: %.5f sec' % timer.times[0])

# 逐列计算A=BC
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
print('逐列计算时间: %.5f sec' % timer.times[1])

# 一次性计算A=BC
timer.start()
A = torch.mm(B, C)
timer.stop()
print('一次性计算时间: %.5f sec' % timer.times[2])

# 生成小批量
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')
# 获取数据集
#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1

# 定义随机梯度下降函数
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()

# 定义训练函数
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初始化模型
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 训练模型
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]

def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

# 使用梯度下降
gd_res = train_sgd(1, 1500, 10)
d2l.plt.show()

# 使用随机梯度下降
# 实验中随机梯度下降的一个迭代轮数耗时更多
#   这是因为随机梯度下降更频繁地更新了参数, 而且一次处理单个观测值效率较低
sgd_res = train_sgd(0.005, 1)
d2l.plt.show()

# 使用小批量随机梯度下降, 批量大小为100
mini1_res = train_sgd(.4, 100)
d2l.plt.show()

# 使用小批量随机梯度下降, 批量大小为10
mini2_res = train_sgd(.05, 10)
d2l.plt.show()

# 比较
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
d2l.plt.show()
