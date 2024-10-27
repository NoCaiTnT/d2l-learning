import torch
import torch.nn as nn
from DotAttention import DotAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_hiddens, dropout, query_size, key_size, value_size):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.attention = DotAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hiddens)
        self.w_k = nn.Linear(key_size, num_hiddens)
        self.w_v = nn.Linear(value_size, num_hiddens)
        self.w_o = nn.Linear(num_hiddens, num_hiddens)

    def forward(self, query, key, value, valid_len):
        query = self.split(self.w_q(query))
        key = self.split(self.w_k(key))
        value = self.split(self.w_v(value))
        if valid_len is not None:
            valid_len = torch.tensor([[i]*self.num_heads for i in valid_len]).reshape(-1)
        output = self.attention(query, key, value, valid_len)

        output = self.merge(output)
        output = self.w_o(output)
        return output

    def split(self, vector):
        vector = vector.reshape(vector.size(0), vector.size(1), self.num_heads, -1)
        vector = vector.permute(0, 2, 1, 3)
        vector = vector.reshape(-1, vector.size(2), vector.size(3))
        return vector

    def merge(self, vector):
        vector = vector.reshape(-1, self.num_heads, vector.size(1), vector.size(2))
        vector = vector.permute(0, 2, 1, 3)
        vector = vector.reshape(vector.size(0), vector.size(1), -1)
        return vector

if __name__ == '__main__':
    num_hiddens = 12
    num_heads = 3
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    multiheadattention = MultiHeadAttention(num_heads, num_hiddens, 0.3, num_hiddens, num_hiddens, num_hiddens)
    outputs = multiheadattention(X, Y, Y, valid_lens)
    print(outputs)