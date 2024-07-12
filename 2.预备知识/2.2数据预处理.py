import os
import pandas as pd
import torch

# 创建人工数据集, 存储在csv文件
# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms, Alley, Price\n')  # 列名
#     f.write('NA, Pave, 127500\n')  # 每行表示一个数据样本
#     f.write('2, NA, 106000\n')
#     f.write('4, NA, 178100\n')
#     f.write('NA, NA, 140000\n')

# 从csv文件加载数据集
data = pd.read_csv(data_file)
print(data)

# 处理缺失的数据, 差值或者删除，一般用插值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())       # 使用均值填充na, 对于数值
print(inputs)

# 对于类别值或者离散值, 将他们视为类别, 扩展列, 对应值为1, 其余为0
inputs = pd.get_dummies(inputs)
print(inputs)

# 转换为张量
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)