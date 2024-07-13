# 注意力分数
#   回顾: f(x) = ∑_i α(x,x_i) y_i = ∑_i^n y_i softmax(-1/2 ((x-x_i))^2)

# 拓展到高维度
#   假设query q∈ℝ^q, m对key-value(k1,v1),..., 这里 k_i∈ℝ^k, v_i∈ℝ^v
#   注意力池化层:
#       f(q,(k1,v1),...,(k_m,v_m)) = ∑_i^m α(q,k_i) v_i  输出维度为v
#       α(q,k) = softmax(a(q,k_i)) = exp(a(q,k_i)) / ∑_{j=1}^m exp(a(q,k_j))  输出为实数

# 加性注意力 Additive Attention
#   可学习参数: W_k∈ℝ^{h×k}, W_q∈ℝ^{h×q}, v∈ℝ^h
#   a(q,k) = v^T tanh(W_q q + W_k k)     (1×h) * [(h×k)(k×1)+(h×q)(q×1)] = (1×h) * (h×1) = 1
#   等价于将key和query合并起来后放入到一个隐藏大小为h, 输出大小为1的单隐藏层MLP中
#   用于query, key, value的长度不相同的情况

# 缩放点积注意力 Scaled Dot-Product Attention
#   如果query和key都是通用的长度 q,k_i∈ℝ^{d}, 那么可以
#       a(q,k_i) = <q,k_i> / sqrt(d)   (1×d) * (d×1) = 1    # 除以sqrt(d)是为了缩放点积, 防止内积过大
#   向量化版本:
#       Q∈ℝ^{n×d}, K∈ℝ^{m×d}, V∈ℝ^{m×v}
#       注意力分数: a(Q,K) = QK^T / sqrt(d)           输出 ∈ ℝ^{n×m}
#       注意力池化: f(Q,K,V) = softmax(a(Q,K)) V      输出 ∈ ℝ^{n×v}

# 总结
#   注意力分数是query和key的相似度, 注意力权重是分数的softmax结果
#   两种常见的分数计算:
#       加性注意力: 将query和key合并起来, 进入一个单输出单隐藏层MLP
#       缩放点积注意力: 直接将query和ket做内积

# 注意力打分函数
import math
import torch
from torch import nn
from d2l import torch as d2l

# 遮蔽softmax操作
# 用于将填充<pad>位置的值设置为很小的值, 使其在softmax后为0
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上遮蔽元素来执行 softmax 操作"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6) # 将填充的位置设置为一个很小的值, 这里为-1e6
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 测试masked_softmax
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

# 加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    # valid_lens: 有效长度, 和query的长度相同, 意思是对于每个query, 应该考虑前多少个(key, value)对
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)       # queries: (batch_size, query的长度(有多少个查询), num_hiddens), keys: (batch_size, 键值对个数, num_hiddens)
        # 在维度扩展后, 沿着最后一个维度广播                          # 中间的维度不同, 不能直接相加
        features = queries.unsqueeze(2) + keys.unsqueeze(1)     # 在第二维度上广播, queries: (batch_size, query的长度, 1, num_hiddens),
                                                                # 在第二维度上广播, keys: (batch_size, 1, 键值对个数, num_hiddens)
                                                                # features: (batch_size, query的长度, 键值对个数, num_hiddens)
        features = torch.tanh(features)
        scores = self.v(features).squeeze(-1)                   # scores: (batch_size, query的长度, 键值对个数)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.attention_weights, values)             # 按batch做乘法 (batch_size, query的长度, 键值对个数) * (batch_size, 键值对个数, value的维度) = (batch_size, query的长度, value的维度)

# 测试AdditiveAttention
# batch-size = 2, query的长度 = 1, query的维度 = 20, key-value的个数 = 10, key的维度 = 2
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.normal(0,1,(2, 10, 2))
#  key-value的个数 = 10, value的维度 = 4
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)    # (2, 10, 4)
# 每个batch中有效的键值对个数
valid_lens = torch.tensor([2, 6])
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
print(attention(queries, keys, values, valid_lens))

# 注意力权重
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()

# 缩放点积注意力
#@save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)     # (batch_size，查询的个数，“键－值”对的个数)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)      # (batch_size，查询的个数，值的维度)

# 测试DotProductAttention
queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
print(attention(queries, keys, values, valid_lens))

# 注意力权重
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()