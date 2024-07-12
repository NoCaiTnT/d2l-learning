import torch
from torch import nn
from torch.nn import functional as F

# 加载和保存张量
# 保存
x = torch.arange(4)
torch.save(x, 'x-file')
# 加载
x2 = torch.load('x-file')
print(x2)

# 存储一个张量列表, 然后读回内存
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print(x2, y2)

# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')

# 为了恢复模型, 实例化原始多层感知机模型的一个备份, 直接读取文件中存储的参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone)

# 验证
Y_clone = clone(X)
print(Y_clone == Y)