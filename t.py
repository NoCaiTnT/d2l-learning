import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, num_hiddens):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len
        self.num_hiddens = num_hiddens
        self.P = self.getPosition()

    def getPosition(self):
        P = torch.zeros(size=(self.max_len, self.num_hiddens))
        pos = torch.arange(0, self.max_len, dtype=torch.float32).reshape(-1, 1)
        div = torch.exp(torch.arange(0, self.num_hiddens, 2) / self.num_hiddens * -1.0 * torch.log(torch.tensor(10000.0,dtype=torch.float32)))

        P[:, 0::2] = torch.sin(pos * div)
        P[:, 1::2] = torch.cos(pos * div)
        return P

    def forward(self, inputs):
        return inputs + self.P.expand(inputs.size(0), -1, -1)

class MaskSoftmax(nn.Module):
    def __init__(self):
        super(MaskSoftmax, self).__init__()

    def forward(self, inputs, valid_len=None):
        if valid_len is None:
            return torch.softmax(inputs, dim=-1)

        mask = torch.zeros(size=(inputs.size(0), inputs.size(1), inputs.size(2)), dtype=torch.float32)
        for b in inputs.size(0):
            inputs[b, :, valid_len[b]] += -1e6

        return torch.softmax(inputs, dim=-1)

class AddAttention(nn.Module):
    def __init__(self, num_hiddens, query_size, key_size, dropout):
        super(AddAttention, self).__init__()
        self.num_hiddens = num_hiddens
        self.w_q = nn.Linear(query_size, num_hiddens)
        self.w_k = nn.Linear(key_size, num_hiddens)
        self.w_v = nn.Linear(num_hiddens, 1)
        self.mask_softmax = MaskSoftmax()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_len=None):
        q = self.w_q(query).unsqueeze(2)
        k = self.w_k(key).unsqueeze(1)
        s = self.w_v(torch.tanh(q+k)).squeeze(-1)
        s = self.dropout(self.mask_softmax(s, valid_len))
        return torch.bmm(s, value)

class DotAttention(nn.Module):
    def __init__(self, dropout):
        super(DotAttention).__init__()
        self.dropout = nn.Dropout(dropout)
        self.mask_softmax = MaskSoftmax()

    def forward(self, query, key, value, valid_len=None):
        k = key.permute(0, 2, 1)
        s = torch.bmm(query, k) / torch.sqrt(torch.tensor(query.size(2)))
        s = self.dropout(self.mask_softmax(s, valid_len))
        return torch.bmm(s, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_hiddens, query_size, key_size, value_size, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.w_q = nn.Linear(query_size, num_hiddens)
        self.w_k = nn.Linear(key_size, num_hiddens)
        self.w_v = nn.Linear(value_size, num_hiddens)
        self.attention = DotAttention(dropout)
        self.w_o = nn.Linear(num_hiddens, num_hiddens)

    def forward(self, query, key, value, valid_len=None):
        q = self.split(self.w_q(query))
        k = self.split(self.w_k(key))
        v = self.split(self.w_v(value))
        if valid_len is not None:
            valid_len = torch.tensor([[i] * self.num_heads for i in valid_len]).reshape(-1)
        r = self.attention(q, k, v, valid_len)
        return self.w_o(self.merge(r))

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

class AddNorm(nn.Module):
    def __init__(self, dorpout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dorpout)
        self.norm = nn.LayerNorm()

    def forward(self, X, Y):
        return self.norm(X + self.dropout(Y))

class FFN(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.linear2 = nn.Linear(ffn_num_hiddens, num_hiddens)

    def forward(self, X):
        return self.linear2(torch.relu(self.linear1(X)))

class EncoderBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, ffn_num_hiddens):
        super(EncoderBlock, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(num_heads, num_hiddens, num_heads, num_heads, num_heads, dropout)
        self.addNorm1 = AddNorm(dropout)
        self.ffn = FFN(num_hiddens, ffn_num_hiddens)
        self.addNorm2 = AddNorm(dropout)

    def forward(self, X, valid_len=None):
        Y = self.multiHeadAttention(X, X, X, valid_len)
        X = self.addNorm1(X, Y)
        Y = self.ffn(X)
        return self.addNorm2(X, Y)

class DecoderBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, ffn_num_hiddens):
        super(DecoderBlock, self).__init__()
        self.multiHeadAttention1 = MultiHeadAttention(num_heads, num_hiddens, num_heads, num_heads, num_heads, dropout)
        self.addNorm1 = AddNorm(dropout)
        self.multiHeadAttention2 = MultiHeadAttention(num_heads, num_hiddens, num_heads, num_heads, num_heads, dropout)
        self.addNorm2 = AddNorm(dropout)
        self.ffn = FFN(num_hiddens, ffn_num_hiddens)
        self.addNorm3 = AddNorm(dropout)

    def forward(self, X, I, X_valid_len=None, Y_valid_len=None):
        Y = self.multiHeadAttention1(X, X, X, X_valid_len)
        X = self.addNorm1(X, Y)
        Y = self.multiHeadAttention2(X, I, I, Y_valid_len)
        X = self.addNorm2(X, Y)
        Y = self.ffn(X)
        return self.addNorm3(X, Y)

class RoPE(nn.Module):
    def __init__(self, num_hiddens, max_len):
        super(RoPE).__init__()
        self.num_hiddens = num_hiddens
        self.max_len = max_len
        self.P = self.getP()

    def getP(self):
        P = torch.zeros(size=(self.max_len, self.num_hiddens))
        pos = torch.arange(0, self.max_len, dtype=torch.float32).reshape(-1, 1)
        div = torch.exp(torch.arange(0, self.num_hiddens, 2, dtype=torch.float32) / self.num_hiddens * -1 * torch.tensor(10000.0, dtype=torch.float32))
        P[:, 0::2] = torch.sin(pos * div)
        P[:, 1::2] = torch.cos(pos * div)
        return P.unsqueeze(0)

    def forward(self, X):
        sin = self.P[:, :X.size(1), 0::2].expand(X.size(0), -1, -1)
        cos = self.P[:, :X.size(1), 1::2].expand(X.size(0), -1, -1)

        X = X * cos + torch.cat((-X[..., 1:2], X[..., 0::2]), dim=-1) * sin
        return X