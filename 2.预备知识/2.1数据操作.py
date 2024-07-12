import torch

# 生成行向量
x = torch.arange(12)
print(x)

# 张量形状
print(x.shape)

# 张量元素个数
print(x.numel())

# 改变张量形状, 但不改变元素数量和值
X = x.reshape(3, 4)
print(X)

# 全0, 全1, 其他常量或从特定分布随机采样
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))

# 通过列表赋值
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# 标准算数运算, 按元素运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)

# 其他按元素运算的方式
print(torch.exp(x))

# 张量的连结(堆叠), 给出堆叠的纬度
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))

# 通过逻辑运算符构建二元张量
print(X == Y)

# 求和所有元素
print(X.sum())

# 广播机制: 张量形状不同, 按元素运算, 需保证维度一样
# 如果遵守以下规则, 则两个tensor是"可广播的":
#   每个tensor至少有一个维度；
#   遍历tensor所有维度时, 从末尾随开始遍历, 两个tensor存在下列情况:
#       tensor维度相等
#       tensor维度不等且其中一个维度为1
#       tensor维度不等且其中一个维度不存在
# 如果两个tensor是"可广播的", 则计算过程遵循下列规则:
#   如果两个tensor的维度不同, 则在维度较小的tensor的前面增加维度, 使它们维度相等
#   对于每个维度, 计算结果的维度值取两个tensor中较大的那个值
#   两个tensor扩展维度的过程是将数值进行复制
a = torch.arange(3).reshape((3, 1))     # 2维, 3×1
b = torch.arange(2).reshape((1, 2))     # 2维, 1×2
print(a)
print(b)
print(a + b)

# 张量元素的访问
print(X[-1])    #最后一行
print(X[1:3])   #第1, 2行, 左闭右开

# 张量的写入
X[1, 2] = 9
print(X)

# 为多个元素赋值
X[0:2, :] = 12
print(X)

# 运行一些操作可能会导致为新结果分配内存
before = id(Y)
Y = Y + X
print(id(Y) == before)

# 执行原地操作
Z = torch.zeros_like(Y)
print('id(Z): ', id(Z))
Z[:] = X + Y
print('id(Z): ', id(Z))

# 或者使用自加:
before = id(X)
X += Y      # 或 X[:] = X + Y
print(id(x) == before)

# 与NumPy的转换
A = X.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))

# 转换为Python标量
a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))
