import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from AddNorm_FFN import AddNorm, FFN

class EncoderBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, ffn_num_hiddens):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, num_hiddens, dropout, num_hiddens, num_hiddens, num_hiddens)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = FFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_len):
        Y = self.attention(X, X, X, valid_len)
        X = self.addnorm1(X, Y)
        Y = self.ffn(X)
        output = self.addnorm2(X, Y)
        return output
