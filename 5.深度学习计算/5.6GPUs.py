import torch
from torch import nn

print(torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:0'))

# 查询gpu数量
print(torch.cuda.device_count())

# 尝试使用第i个gpu
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# 返回所有gpu
def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())

# 查询张量所在的设备
x = torch.tensor([1, 2, 3])
print(x.device)

# 创建时放入gpu上
X = torch.ones(2, 3, device=try_gpu())
print(X)

# 在第2个gpu上创建张量
Y = torch.rand(2, 3, device=try_gpu(1))
print(Y)

# 计算时, 计算的张量必须在同一个gpu/cpu上
Z = Y.cuda(0)   # 将Y的值复制到gpu: 0上
print(Z)

# 在同一个gpu上进行计算
print(X + Z)

# 当张量已经在gpu上, 使用cuda()会返回自己
print(Z.cuda(0) is Z)

# 神经网络与gpu
# 将模型参数放在GPU上
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())      # 将参数复制到gpu: 0上
print(net(X))

# 确认模型参数存储在同一个gpu上
print(net[0].weight.data.device)

# 不经意地移动数据可能会显著降低性能(数据在设备(CPU, GPU和其他机器)之间传输数据比计算慢得多)
# 一个典型的错误如下:
#   计算GPU上每个小批量的损失, 并在命令行中将其报告给用户(或将其记录在NumPy ndarray中)时, 将触发全局解释器锁, 从而使所有GPU阻塞, 最好是为GPU内部的日志分配内存, 并且只移动较大的日志
# nn.module使用.to()移动到gpu上
# 张量使用.cuda()移动到gpu上
