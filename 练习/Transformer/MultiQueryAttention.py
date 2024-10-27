import torch
import torch.nn as nn
from DotAttention import DotAttention

class MultiQueryAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, query_size, key_size, value_size, dropout):
        super(MultiQueryAttention, self).__init__()
        self.w_q = nn.Linear(query_size, num_hiddens)
        self.w_k = nn.Linear(key_size, num_hiddens // num_heads)
        self.w_v = nn.Linear(value_size, num_hiddens // num_heads)
        self.w_o = nn.Linear(num_hiddens, num_hiddens)
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.attention = DotAttention(dropout)

    def forward(self, q, k, v, valid_len=None):
        q = self.split(self.w_q(q))
        k = self.share(self.w_k(k))
        v = self.share(self.w_v(v))
        if valid_len is not None:
            valid_len = torch.tensor([[i] * self.num_heads for i in valid_len]).reshape(-1)
        r = self.attention(q, k, v, valid_len)
        r = self.merge(r)
        return self.w_o(r)

    def share(self, X):
        X = X.unsqueeze(1)
        X = X.expand(-1, self.num_heads, -1, -1)
        X = X.reshape(-1, X.size(2), X.size(3))
        return X

    def split(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(-1, X.size(2), X.size(3))
        return X

    def merge(self, X):
        X = X.reshape(-1, self.num_heads, X.size(1), X.size(2))
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(X.size(0), X.size(1), -1)
        return X

# 参数设定
batch_size = 2
num_queries = 4
num_keys = 4
num_hiddens = 8
num_heads = 4
query_size = 6
key_size = 6
value_size = 6
dropout = 0.1
num_groups = 2

# 创建GroupQueryAttention实例
attention = MultiQueryAttention(num_hiddens, num_heads, query_size, key_size, value_size, dropout)

# 生成随机的查询、键和值张量
q = torch.rand((batch_size, num_queries, query_size))
k = torch.rand((batch_size, num_keys, key_size))
v = torch.rand((batch_size, num_keys, value_size))

# 执行前向传播
output = attention(q, k, v)

# 打印输出结果
print("Output shape:", output.shape)
print("Output:", output)
