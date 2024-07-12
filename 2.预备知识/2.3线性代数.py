import torch

# 标量: 由只有一个元素的张量表示
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y, x * y, x / y, x ** y)

# 向量: 标量值组成的列表
x = torch.arange(4)
print(x)

# 通过索引访问任一元素
print(x[3])

# 张量的长度
print(len(x))

# 形状
print(x.shape)

# 创建m×n的矩阵
A = torch.arange(20).reshape(5, 4)
print(A)

# 矩阵的转置
print(A.T)

# 对称矩阵, 其转置等于本身
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

# 构建任意维度的数据结构
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 形状相同的两个张量, 按元素二元运算的结果都是相同形状的张量
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A)
print(A + B)

# 按元素乘法, 称为哈达玛积(Hadamard product)
print(A * B)
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)

# 计算张量元素和
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())
A = torch.arange(20).reshape(5, 4)
print(A.shape, A.sum())

# 按张量的轴求和
A = torch.arange(40).reshape(2, 5, 4)
print(A)
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)

print(A.sum(axis=[0, 1]))

# 求均值
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.mean(), A.sum() / A.numel())
print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

# 计算总和或均值时保持维度不变, 即不丢掉维度为1的维度
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

# 通过广播将A除以sum_A
print(A / sum_A)

# 按某个轴累加求和
print(A.cumsum(axis=0))

# 点积: 相同位置的按元素乘积求和
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))
# 等价于按元素乘法，然后求和
print(torch.sum(x * y))

# 向量点积
print(A.shape, x.shape, torch.mv(A, x))

# 矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))

# 向量范数, L2范数: 向量的长度, 元素平方和的平方根
u = torch.tensor(([3.0, -4.0]))
print(torch.norm(u))

# 向量范数, L1范数: 元素绝对值之和
print(torch.abs(u).sum())

# 矩阵范数, F范数(弗罗贝尼乌斯范数, Frobenius norm): 矩阵元素的平方和的平方根
print(torch.norm(torch.ones((4, 9))))
