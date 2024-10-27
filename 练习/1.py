import torch

# 假设序列长度为5
seq_length = 5

# 生成上三角矩阵，diagonal=1 表示对角线以上的部分为1，对角线及其以下为0
upper_triangular_matrix = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1)

print(upper_triangular_matrix)

# 将上三角矩阵转换为布尔类型，并反转它
causal_mask = upper_triangular_matrix == 0  # 将上三角部分转换为 False, 下三角部分和对角线转换为 True

# 假设有一个注意力分数张量 (batch_size, num_heads, seq_length, seq_length)
scores = torch.randn(3, 8, seq_length, seq_length)

# 使用上三角掩码
scores = scores.masked_fill(causal_mask == False, float('-inf'))

# 计算 softmax 以获得注意力权重
attention_weights = torch.nn.functional.softmax(scores, dim=-1)

print(attention_weights)
