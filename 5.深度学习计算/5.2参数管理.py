import torch
from torch import nn

# 单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

# 参数访问 访问的是第2层的权重, 即nn.Linear(8, 1)的权重
print(net[2].state_dict())

# 直接访问目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].bias.data.item())
# 访问梯度
print(net[2].weight.grad == None)

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # *号是解包操作, 一般对元组或列表使用, 变成单独的元素
print(*[(name, param.shape) for name, param in net.named_parameters()])
# 或者使用名字直接获取参数
print(net.state_dict()['2.bias'].data)

# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())  # 可以给层传名字
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
# 输出网络模型
print((rgnet))

# 参数初始化
# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)         # 带下划线为替换函数, 会直接将其替换掉
        nn.init.zeros_(m.bias)
net.apply(init_normal)                              # 对net里面所有层循环调用init_normal函数
print(net[0].weight.data[0], net[0].bias.data[0])
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])

#对某些块应用不同的初始化方法
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)   # 使用Xavier均值分布
        # nn.init.xavier_normal_(m.weight)  # 使用Xavier正态分布
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5   # 保留绝对值大于5的权重(大于5后面为1, *=1即为自身, 繁殖后面为0, *=0为0)

net.apply(my_init)
print(net[0].weight[:2])

# 直接设置参数
# print(net[0])
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])

# 参数绑定
# 在多个层间共享参数: 定义一个稠密层，然后使用它的参数来设置另一个层的参数
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
print(net(X))
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])