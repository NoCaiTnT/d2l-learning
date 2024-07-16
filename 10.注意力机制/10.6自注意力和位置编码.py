# 自注意力
#   给定序列x_1, ..., x_n, 任意x_i ∈ ℝ^d
#   自注意力池化层将x_i当做key, value, query来对序列抽取特征得到y_1, ..., y_n
#       y_i = f(x_i, (x_1,x_1),...,(x_n,x_n)) ∈ ℝ^d

# 跟CNN, RNN的对比
#               CNN         RNN         自注意力
#   计算复杂度 O(k n d^2)   O(n d^2)      O(n^2 d)
#   并行度       O(n)        O(1)         O(n)
#   最长路径    O(n/k)       O(n)         O(1)

# 位置编码
#   跟CNN/RNN不同, 自注意力并没有记录位置信息
#   位置编码将位置信息注入到输入里
#       假设长度为n的序列是X ∈ ℝ^(n×d), 那么使用位置编码矩阵P ∈ ℝ^(n×d)来输出X+P作为自编码输入
#   P的元素如下计算:
#       P_{i,2j} = sin(i/10000^(2j/d)), P_{i,2j+1} = cos(i/10000^(2j/d))

# 绝对位置信息
#   计算机使用的二进制编码   可以看到第一列变化最慢, 最后一列变化最快
#       0: 0000          将每一列看成一个样本, 每一行看成一个特征
#       1: 0001          给序列中的每一个样本 加上一个 独一无二的长为 d 的位置信息, 作为输入
#       2: 0010
#       3: 0011
#       4: 0100
#       5: 0101
#       6: 0110
#       7: 0111
#       8: 1000

# 相对位置信息
#   位置与i+δ处的位置编码可以线性投影到位置i处的位置编码来表示
#   记w_j = 1/10000^(2j/d), δ为偏移量, 那么
#       ┌ cos(δw_j)  sin(δw_j) ┐ ┌  p_{i,2j}  ┐ = ┌  p_{i+δ,2j}  ┐
#       └ -sin(δw_j) cos(δw_j) ┘ └ p_{i,2j+1} ┘   └ p_{i+δ,2j+1} ┘
#                   ↑
#              投影矩阵跟i无关
#   意思就是, 对于一个句子, 第一个词到第三个词的距离, 和第二个词到第四个词的距离, 他们都可以通过一个相同的投影矩阵来进行线性变换

# 总结
#   自注意力池化层将x_i当做key, value, query来对序列抽取特征
#   完全并行, 最长序列为1, 但对长序列计算复杂度高
#   位置编码在输入中加入位置信息, 使得自注意力能够记忆位置信息

# 代码
import math
import torch
from torch import nn
from d2l import torch as d2l

num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
print(attention(X, X, X, valid_lens).shape)

# 位置编码
#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# 行代表标记在序列中的位置, 列代表位置编码的不同维度
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
d2l.plt.show()

# 绝对位置信息
for i in range(8):
    print(f'{i}的二进制是：{i:>03b}')

P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
d2l.plt.show()

