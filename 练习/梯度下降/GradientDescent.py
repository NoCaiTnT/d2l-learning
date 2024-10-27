import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化参数
# w = nn.Parameter(torch.randn(size=(1, 1), device=device))
# b = nn.Parameter(torch.zeros(size=(1,), device=device))

w = torch.randn(size=(1, 1), device=device)
b = torch.zeros(size=(1,), device=device)

# 输入和目标
inputs = torch.tensor([i for i in range(10)], dtype=torch.float32, device=device).reshape(-1, 1)
targets = torch.tensor([i * 2 + 3 for i in range(10)], dtype=torch.float32, device=device).reshape(-1, 1)

epochs = 100
lr = 0.01


# 自定义均方误差损失函数
def mseloss(y, target):
    return ((y - target) ** 2).mean()

# 手动更新参数的训练循环
for epoch in range(epochs):
    # 前向传播
    y = torch.mm(inputs, w) + b

    # 计算损失
    loss = mseloss(y, targets)

    # 手动计算梯度
    dloss2y = 2 * (y - targets) / len(inputs)
    dy2w = torch.mm(inputs.t(), dloss2y)
    dy2b = dloss2y.sum()

    # 更新参数
    # with torch.no_grad():
    w -= lr * dy2w
    b -= lr * dy2b

    # 打印每个epoch的损失
    print(f'epoch: {epoch}, loss: {loss.item()}')

def predict(x):
    return torch.mm(x, w) + b

x = torch.tensor([[1], [-10]], dtype=torch.float32, device=device)
print(predict(x))
print(w, b)